//! Rate-distortion optimised quantisation and sign data hiding for the
//! H.265 encoder.
//!
//! # What the quantiser chooses
//!
//! A dead-zone quantiser rounds every coefficient on its own. Whether a
//! level is worth its bits depends on the rest of the block — a level of
//! one after a long run of zeros costs a significance flag, a sign and,
//! if it moves the last position, every significance flag between — so
//! the levels are chosen here jointly, in the three steps the literature
//! on rate-distortion optimised quantisation describes (per coefficient,
//! per coefficient group, then the last position):
//!
//! 1. **Per coefficient**, in the order the writer codes them (sub-blocks
//!    from the last one back, positions within each from high to low
//!    frequency): the rounded level, one less, and zero where the rounded
//!    level is at most two, each priced at its distortion plus lambda
//!    times its bits, the cheapest kept. The bits follow the writer's
//!    context derivation as it would run over the levels already chosen:
//!    the significance context of the position, the greater-than-one
//!    context set and its counter, the greater-than-two flag of the
//!    first such level, the Rice parameter of the remainders, one bypass
//!    bit per sign.
//! 2. **Per coefficient group**: a group between the first and the last
//!    whose levels cost more than its distortion with every level zero
//!    plus a `coded_sub_block_flag` of 0 is dropped.
//! 3. **The last position**: every significant position is costed as the
//!    last one — its last-position prefix and suffix bits, the chosen
//!    costs before it (the `coded_sub_block_flag` of every group between
//!    the first and its own included, an empty group's significance flags
//!    not), the zero distortion after it — and the cheapest kept, unless
//!    coding nothing at all (a clear coded block flag, where the caller
//!    names its context) is cheaper still. Without that last test a
//!    block whose one significant coefficient sits just above half a
//!    step was always coded, which is what made RDOQ lose on inter
//!    chroma until round 5.
//!
//! The distortion is measured in the transform domain and converted to the
//! sample domain by the transform's scale: the HEVC forward transform
//! leaves an orthonormal coefficient multiplied by `2^(15 - BitDepth -
//! log2(N))`, so a coefficient error `e` is a sample-domain SSD of `e^2 /
//! 2^(2 * (15 - BitDepth - log2(N)))` (Parseval, up to the rounding the
//! integer transforms keep). The bits are context-coded bins priced at
//! the probability states of the slice's initial contexts
//! (`cabac_enc::state_bits`), as every other rate in this encoder.
//!
//! # Sign data hiding
//!
//! Under the PPS flag a sub-block whose first and last significant scan
//! positions are more than three apart does not code the sign of its first
//! (lowest-frequency) coefficient: the parity of the sum of its levels
//! does, odd meaning negative. Where the levels have the wrong parity one
//! of them moves by one — the change whose distortion grows least, never
//! the change that would empty the sub-block's first or the block's last
//! position — which costs far less than the sign bit it saves on average.
//! Where the block carries a Lagrangian the move's rough change in bins
//! counts too. [`hide_signs`] does that after either quantiser.

use crate::cabac_enc::state_bits;
use crate::hevc::ctx::{
    COEFF_ABS_LEVEL_GREATER1_FLAG_OFFSET, COEFF_ABS_LEVEL_GREATER2_FLAG_OFFSET, Contexts,
    LAST_SIGNIFICANT_COEFF_X_PREFIX_OFFSET, LAST_SIGNIFICANT_COEFF_Y_PREFIX_OFFSET,
    SIGNIFICANT_COEFF_GROUP_FLAG_OFFSET,
};
use crate::hevc::residual::{last_sig_ctx, scan_pos, sig_ctx_of, sub_block_tables};

/// One transform block's quantisation problem.
pub(crate) struct Block<'a> {
    /// log2 of the block's edge, 2 to 5.
    pub log2: u32,
    /// 0 luma, 1 or 2 chroma.
    pub c_idx: usize,
    /// The block's scan (`residual_scan_idx`).
    pub scan_idx: u32,
    /// `Qp'` of the block, the bit-depth offset included.
    pub qp: i32,
    /// Sample bit depth.
    pub bit_depth: u32,
    /// The SSD Lagrangian.
    pub lambda: f64,
    /// The contexts the bins are priced at.
    pub cx: &'a Contexts,
    /// Whether the PPS hides signs.
    pub sign_hiding: bool,
    /// The context of the block's coded block flag, where the block may
    /// be left empty: the levels are then weighed against coding nothing.
    pub cbf_ctx: Option<usize>,
}

/// What one coefficient position contributes, as decided.
#[derive(Clone, Copy, Default)]
struct Pos {
    /// The raster index of the position.
    idx: usize,
    /// `|c| * scale`, the coefficient on the quantiser's scale.
    scaled: i64,
    /// The level chosen.
    level: i32,
    /// Cost with the level chosen, its significance flag included.
    coded: f64,
    /// Cost with the level chosen and no significance flag: the cost
    /// when this is the last position.
    as_last: f64,
    /// Distortion with the level zero (beyond the last position nothing
    /// is coded at all).
    zero: f64,
    /// The decided cost of every position ahead of this one.
    before: f64,
}

/// Bits of `coeff_abs_level_remaining` `v` at Rice parameter `rice`, as
/// the writer binarises it.
#[inline]
fn remaining_bits(rice: u32, v: u32) -> f64 {
    let prefix = v >> rice;
    if prefix < 4 {
        f64::from(prefix + 1 + rice)
    } else {
        let mut rem = v - (4 << rice);
        let mut k = rice + 1;
        let mut bits = 4u32;
        while rem >= (1 << k) {
            bits += 1;
            rem -= 1 << k;
            k += 1;
        }
        f64::from(bits + 1 + k)
    }
}

/// `last_sig_coeff_{x,y}_prefix` of a coordinate `v`.
#[inline]
fn last_prefix(v: u32) -> u32 {
    if v <= 3 {
        return v;
    }
    let msb = 31 - v.leading_zeros();
    2 * msb + ((v - (1 << msb) >= (1 << (msb - 1))) as u32)
}

/// Bits of one coordinate of the last-position syntax, by its prefix
/// ([`last_prefix`]): the prefix's bins at the contexts from `base`, and
/// its suffix.
fn last_axis_bits(cx: &Contexts, c_idx: usize, log2: u32, base: usize) -> [f64; 10] {
    let (off, shift) = last_sig_ctx(c_idx, log2);
    let c_max = (log2 << 1) - 1;
    let mut t = [0f64; 10];
    let mut run = 0f64;
    for p in 0..=c_max {
        let ctx = cx.c[base + (off + (p >> shift)) as usize];
        let mut bits = run;
        if p < c_max {
            bits += f64::from(state_bits(ctx, 0));
        }
        if p > 3 {
            bits += f64::from((p >> 1) - 1);
        }
        t[p as usize] = bits;
        run += f64::from(state_bits(ctx, 1));
    }
    t
}

/// The greater-than-one and Rice state the writer carries through a
/// sub-block, as levels are added to it in coding order.
#[derive(Clone, Copy)]
struct LevelState {
    /// The sub-block's `ctxSet`.
    ctx_set: u32,
    /// `greater1Ctx`.
    c1: u32,
    /// Significant levels so far in this sub-block.
    count: usize,
    /// Whether a level above one has been seen among the first eight.
    g1_seen: bool,
    /// The Rice parameter for the next remainder, and whether one has
    /// been coded yet.
    rice: u32,
    remaining_seen: bool,
}

impl LevelState {
    fn new(ctx_set: u32) -> Self {
        LevelState {
            ctx_set,
            c1: 1,
            count: 0,
            g1_seen: false,
            rice: 0,
            remaining_seen: false,
        }
    }

    /// Bits of level `l` (at least 1) coded next, sign included, and the
    /// state after it.
    fn price(&self, cx: &Contexts, c_idx: usize, l: i32, sign: bool) -> (f64, LevelState) {
        let mut st = *self;
        let mut bits = if sign { 1.0 } else { 0.0 };
        let k = st.count;
        let mut base = 1;
        let mut threshold = 1;
        if k < 8 {
            let g1_base = COEFF_ABS_LEVEL_GREATER1_FLAG_OFFSET
                + (st.ctx_set * 4) as usize
                + if c_idx > 0 { 16 } else { 0 };
            let g1 = l > 1;
            bits += f64::from(state_bits(cx.c[g1_base + st.c1.min(3) as usize], g1 as u32));
            if st.c1 > 0 {
                st.c1 = if g1 { 0 } else { st.c1 + 1 };
            }
            base = 1 + g1 as i32;
            threshold = 2;
            if g1 && !st.g1_seen {
                st.g1_seen = true;
                let inc = st.ctx_set as usize + if c_idx > 0 { 4 } else { 0 };
                bits += f64::from(state_bits(
                    cx.c[COEFF_ABS_LEVEL_GREATER2_FLAG_OFFSET + inc],
                    (l > 2) as u32,
                ));
                base += (l > 2) as i32;
                threshold = 3;
            }
        }
        if base == threshold {
            let rice = if st.remaining_seen { st.rice } else { 0 };
            bits += remaining_bits(rice, (l - base) as u32);
            st.rice = (rice + (l > 3 * (1 << rice)) as u32).min(4);
            st.remaining_seen = true;
        }
        st.count += 1;
        (bits, st)
    }
}

/// Quantise `coeffs` (the forward transform's output, raster) into
/// `levels` by rate-distortion optimisation, then hide signs if the PPS
/// says so. Returns the number of nonzero levels.
pub(crate) fn quantise(b: &Block<'_>, coeffs: &[i16], levels: &mut [i16]) -> u32 {
    // Positions in coding order's reverse: scan order, sub-block-major.
    // The scratch is per thread and reused: a block is priced thousands of
    // times a picture.
    SCRATCH.with_borrow_mut(|pos| quantise_in(b, coeffs, levels, pos))
}

thread_local! {
    static SCRATCH: std::cell::RefCell<Vec<Pos>> =
        std::cell::RefCell::new(vec![Pos::default(); 1024]);
}

fn quantise_in(b: &Block<'_>, coeffs: &[i16], levels: &mut [i16], pos: &mut [Pos]) -> u32 {
    let n = 1usize << b.log2;
    let log2_sb = b.log2 - 2;
    let sb_w = 1usize << log2_sb;
    let scale = i64::from(crate::dsp::hevc_enc::quant_scale((b.qp % 6) as usize));
    let qbits = crate::dsp::hevc_enc::qbits(b.qp, b.log2, b.bit_depth);
    let step = 1i64 << qbits;
    let tshift = 15 - b.bit_depth as i32 - b.log2 as i32;
    let err_scale = 1.0 / (scale as f64 * scale as f64) * (-2.0 * f64::from(tshift)).exp2();
    let lambda = b.lambda;
    let dist = |scaled: i64, l: i32| -> f64 {
        let e = (scaled - i64::from(l) * step) as f64;
        e * e * err_scale
    };
    let tabs = sub_block_tables();
    let half = step >> 1;
    levels[..n * n].fill(0);
    // Nothing reaches half a step: every level rounds to zero, and the
    // search below could only confirm it.
    let peak = coeffs[..n * n]
        .iter()
        .map(|c| c.unsigned_abs())
        .max()
        .unwrap_or(0);
    if i64::from(peak) * scale < half {
        return 0;
    }
    let raster = raster_of(b.scan_idx, b.log2);
    let Some(last) = raster
        .iter()
        .rposition(|&r| i64::from(coeffs[r as usize].unsigned_abs()) * scale >= half)
    else {
        return 0;
    };
    for (k, &r) in raster[..=last].iter().enumerate() {
        let idx = r as usize;
        let scaled = i64::from(coeffs[idx].unsigned_abs()) * scale;
        pos[k] = Pos {
            idx,
            scaled,
            zero: 0.0,
            ..Pos::default()
        };
    }
    for p in &mut pos[..=last] {
        p.zero = dist(p.scaled, 0);
    }
    let last_sb = last / 16;

    // The sub-blocks the rounded levels code, for the significance
    // contexts' neighbour pattern.
    let mut csbf = [0u8; 64];
    for (sb, chunk) in pos[..=last].chunks(16).enumerate() {
        if chunk.iter().any(|p| (p.scaled + (step >> 1)) >> qbits > 0) {
            let (xs, ys) = scan_pos(b.scan_idx, log2_sb, sb);
            csbf[ys * sb_w + xs] = 1;
        }
    }

    // 1. Per coefficient, in coding order.
    let mut c1_after_prev: Option<u32> = None;
    let mut sb_cost_coded = [0f64; 64];
    let mut sb_cost_zero = [0f64; 64];
    for sb in (0..=last_sb).rev() {
        let (xs, ys) = scan_pos(b.scan_idx, log2_sb, sb);
        let mut prev_csbf = 0usize;
        if xs < sb_w - 1 {
            prev_csbf += csbf[ys * sb_w + xs + 1] as usize;
        }
        if ys < sb_w - 1 {
            prev_csbf += (csbf[(ys + 1) * sb_w + xs] as usize) << 1;
        }
        let mut ctx_of = [0u16; 16];
        sig_ctx_of(
            tabs,
            b.scan_idx,
            b.log2,
            b.c_idx,
            xs,
            ys,
            prev_csbf,
            None,
            16,
            &mut ctx_of,
        );
        let mut ctx_set: u32 = if sb == 0 || b.c_idx > 0 { 0 } else { 2 };
        if c1_after_prev == Some(0) {
            ctx_set += 1;
        }
        let mut st = LevelState::new(ctx_set);
        let top = if sb == last_sb { last % 16 } else { 15 };
        for p in (0..=top).rev() {
            let k = sb * 16 + p;
            let at = pos[k];
            let is_last = k == last;
            let sig_ctx = cx_state(b.cx, ctx_of[p] as usize);
            let sig0 = f64::from(state_bits(sig_ctx, 0));
            let sig1 = f64::from(state_bits(sig_ctx, 1));
            let rounded = ((at.scaled + (step >> 1)) >> qbits) as i32;
            let mut best_level = 0;
            let mut best = at.zero + lambda * sig0;
            let mut best_last = f64::INFINITY;
            let mut best_state = st;
            let lo = if rounded <= 2 { 0 } else { rounded - 1 };
            for l in (lo.max(1)..=rounded).rev() {
                let (bits, after) = st.price(b.cx, b.c_idx, l, true);
                let d = dist(at.scaled, l);
                let with_sig = d + lambda * (bits + sig1);
                let as_last = d + lambda * bits;
                let c = if is_last { as_last } else { with_sig };
                if c < best || (is_last && best_level == 0) {
                    best = c;
                    best_level = l;
                    best_state = after;
                    best_last = as_last;
                }
            }
            if is_last && best_level == 0 {
                best_level = 1;
                let (bits, after) = st.price(b.cx, b.c_idx, 1, true);
                best_state = after;
                best_last = dist(at.scaled, 1) + lambda * bits;
                best = best_last;
            }
            if best_level > 0 {
                st = best_state;
            }
            pos[k].level = best_level;
            pos[k].coded = if best_level > 0 {
                best_last + lambda * sig1
            } else {
                best
            };
            pos[k].as_last = best_last;
            sb_cost_coded[sb] += if best_level > 0 {
                best
            } else {
                at.zero + lambda * sig0
            };
            sb_cost_zero[sb] += at.zero;
        }
        c1_after_prev = if st.count > 0 {
            Some(st.c1)
        } else {
            c1_after_prev
        };
    }

    // 2. Per coefficient group, between the first and the last. A group
    // left empty codes a clear flag and none of its significance flags.
    let mut sb_flag = [[0f64; 2]; 64];
    for sb in 1..last_sb {
        if pos[sb * 16..sb * 16 + 16].iter().all(|p| p.level == 0) {
            for p in &mut pos[sb * 16..sb * 16 + 16] {
                p.coded = p.zero;
            }
        }
        let (xs, ys) = scan_pos(b.scan_idx, log2_sb, sb);
        let mut inc = 0usize;
        if xs < sb_w - 1 {
            inc += csbf[ys * sb_w + xs + 1] as usize;
        }
        if ys < sb_w - 1 {
            inc += csbf[(ys + 1) * sb_w + xs] as usize;
        }
        let inc = if b.c_idx == 0 {
            inc.min(1)
        } else {
            2 + inc.min(1)
        };
        let flag = cx_state(b.cx, SIGNIFICANT_COEFF_GROUP_FLAG_OFFSET + inc);
        sb_flag[sb] = [
            lambda * f64::from(state_bits(flag, 0)),
            lambda * f64::from(state_bits(flag, 1)),
        ];
        if pos[sb * 16..sb * 16 + 16].iter().all(|p| p.level == 0) {
            continue;
        }
        let keep = sb_cost_coded[sb] + lambda * f64::from(state_bits(flag, 1));
        let drop = sb_cost_zero[sb] + lambda * f64::from(state_bits(flag, 0));
        if drop < keep {
            for p in &mut pos[sb * 16..sb * 16 + 16] {
                p.level = 0;
                p.coded = p.zero;
            }
        }
    }

    let mut sb_any = [false; 64];
    for (sb, a) in sb_any.iter_mut().enumerate().take(last_sb + 1) {
        *a = pos[sb * 16..(sb * 16 + 16).min(last + 1)]
            .iter()
            .any(|p| p.level != 0);
    }

    // 3. The last position. `before` is the cost of everything ahead of a
    // candidate, as decided; `after` the distortion of everything beyond.
    let mut run = 0f64;
    for (k, p) in pos[..=last].iter_mut().enumerate() {
        // Entering sub-block `s`: the flag of the one before it, which a
        // last position here or beyond codes (the first's is inferred).
        if k % 16 == 0 && k >= 32 {
            let s = k / 16 - 1;
            run += sb_flag[s][usize::from(sb_any[s])];
        }
        p.before = run;
        run += p.coded;
    }
    // The last-position syntax prices x and y apart: a table of each.
    let last_x = last_axis_bits(
        b.cx,
        b.c_idx,
        b.log2,
        LAST_SIGNIFICANT_COEFF_X_PREFIX_OFFSET,
    );
    let last_y = last_axis_bits(
        b.cx,
        b.c_idx,
        b.log2,
        LAST_SIGNIFICANT_COEFF_Y_PREFIX_OFFSET,
    );
    let mut after = 0f64;
    let mut best_last = last;
    let mut best_cost = f64::INFINITY;
    let cbf1 = b.cbf_ctx.map_or(0.0, |c| {
        lambda * f64::from(state_bits(cx_state(b.cx, c), 1))
    });
    for k in (0..=last).rev() {
        let p = &pos[k];
        if p.level > 0 {
            let (xc, yc) = (p.idx & (n - 1), p.idx >> b.log2);
            let (lx, ly) = if b.scan_idx == 2 {
                (yc as u32, xc as u32)
            } else {
                (xc as u32, yc as u32)
            };
            let cost = p.before
                + p.as_last
                + after
                + cbf1
                + lambda * (last_x[last_prefix(lx) as usize] + last_y[last_prefix(ly) as usize]);
            if cost < best_cost {
                best_cost = cost;
                best_last = k;
            }
        }
        after += p.zero;
    }
    // Nothing coded at all: every position's zero distortion and a clear
    // coded block flag.
    if let Some(c) = b.cbf_ctx
        && after + lambda * f64::from(state_bits(cx_state(b.cx, c), 0)) <= best_cost
    {
        return 0;
    }
    let mut nz = 0u32;
    for p in pos.iter().take(best_last + 1) {
        if p.level > 0 {
            levels[p.idx] = if coeffs[p.idx] < 0 { -p.level } else { p.level } as i16;
            nz += 1;
        }
    }
    if b.sign_hiding && nz > 1 {
        nz = hide_signs_at(b, coeffs, levels, nz, best_last);
    }
    nz
}

/// The raster index of every scan position of a block, sub-block by
/// sub-block (`[scan_idx][log2 - 2]`), built once.
fn raster_of(scan_idx: u32, log2: u32) -> &'static [u16] {
    static T: std::sync::OnceLock<Vec<Vec<Vec<u16>>>> = std::sync::OnceLock::new();
    let t = T.get_or_init(|| {
        let tabs = sub_block_tables();
        (0..3u32)
            .map(|scan| {
                (2..=5u32)
                    .map(|log2| {
                        let n = 1usize << log2;
                        let log2_sb = log2 - 2;
                        let mut v = vec![0u16; n * n];
                        for sb in 0..1usize << (2 * log2_sb) {
                            let (xs, ys) = scan_pos(scan, log2_sb, sb);
                            for p in 0..16 {
                                let (xp, yp) = tabs.pos[scan as usize][p];
                                v[sb * 16 + p] =
                                    (((ys << 2) + yp as usize) * n + (xs << 2) + xp as usize)
                                        as u16;
                            }
                        }
                        v
                    })
                    .collect()
            })
            .collect()
    });
    &t[scan_idx as usize][(log2 - 2) as usize]
}

#[inline]
fn cx_state(cx: &Contexts, i: usize) -> u8 {
    cx.c[i]
}

/// Sign data hiding over levels already chosen (by either quantiser):
/// every sub-block the rule applies to whose level sum has the wrong
/// parity for its first coefficient's sign gets one level moved by one,
/// the move that adds least distortion. Returns the nonzero count.
pub(crate) fn hide_signs(b: &Block<'_>, coeffs: &[i16], levels: &mut [i16]) -> u32 {
    let n = 1usize << b.log2;
    let nz = levels[..n * n].iter().filter(|&&v| v != 0).count() as u32;
    // A sub-block hides a sign only with two significant coefficients at
    // least four scan positions apart.
    if nz < 2 {
        return nz;
    }
    let n_sb = 1usize << (2 * (b.log2 - 2));
    let idx = raster_of(b.scan_idx, b.log2);
    // The block's last significant position, which no move may empty.
    let Some(last_k) = (0..n_sb * 16).rev().find(|&k| levels[idx[k] as usize] != 0) else {
        return 0;
    };
    hide_signs_at(b, coeffs, levels, nz, last_k)
}

/// [`hide_signs`] for levels whose nonzero count `nz` (at least 2) and
/// last significant scan position `last_k` the caller knows.
fn hide_signs_at(b: &Block<'_>, coeffs: &[i16], levels: &mut [i16], nz: u32, last_k: usize) -> u32 {
    let scale = i64::from(crate::dsp::hevc_enc::quant_scale((b.qp % 6) as usize));
    let qbits = crate::dsp::hevc_enc::qbits(b.qp, b.log2, b.bit_depth);
    let step = 1i64 << qbits;
    let idx = raster_of(b.scan_idx, b.log2);
    let at = |k: usize| idx[k] as usize;
    let mut nz = nz;
    // A move's rate, roughly: the bins a level gains or sheds when it
    // crosses zero, one and two, weighed against its distortion in the
    // sample domain wherever the caller gives a Lagrangian. Weighed at 0.6
    // of it (round 5, 2026-11-15, x265 parity at slow): -5.51% -> -5.85%
    // BD-rate, ducks +0.43% -> -0.03%; at 1.0 -5.85%, at 1.6 -5.51%, at
    // 0.35 -5.89%.
    let tshift = 15 - b.bit_depth as i32 - b.log2 as i32;
    let es = 1.0 / (scale as f64 * scale as f64) * (-2.0 * f64::from(tshift)).exp2();
    const RATE_WEIGHT: f64 = 0.6;
    let bits_up = |l: i32| -> f64 {
        match l {
            0 => 3.0,
            1 => 1.0,
            _ => 0.5,
        }
    };
    for sb in 0..=last_k / 16 {
        let base = sb * 16;
        let mut first = None;
        let mut lastp = None;
        let mut sum = 0i32;
        for p in 0..16 {
            let l = levels[at(base + p)];
            if l != 0 {
                first.get_or_insert(p);
                lastp = Some(p);
                sum += i32::from(l.unsigned_abs());
            }
        }
        let (Some(first), Some(lastp)) = (first, lastp) else {
            continue;
        };
        if lastp - first <= 3 {
            continue;
        }
        let neg = levels[at(base + first)] < 0;
        if (sum % 2 == 1) == neg {
            continue;
        }
        // Move one level by one: the move whose distortion grows least. A
        // move may not empty the sub-block's first coefficient or the
        // block's last, nor create a coefficient beyond the last.
        let mut best: Option<(i64, usize, i32)> = None;
        let top = if sb == last_k / 16 { last_k % 16 } else { 15 };
        for p in first..=top {
            let i = at(base + p);
            let l = i32::from(levels[i].unsigned_abs());
            let scaled = i64::from(coeffs[i].unsigned_abs()) * scale;
            let err = |lv: i32| {
                let e = scaled - i64::from(lv) * step;
                e * e
            };
            for delta in [1i32, -1] {
                let nl = l + delta;
                if nl < 0 || (nl == 0 && (p == first || base + p == last_k)) {
                    continue;
                }
                let mut cost = err(nl) - err(l);
                if b.lambda > 0.0 {
                    let db = if delta > 0 { bits_up(l) } else { -bits_up(nl) };
                    cost += ((b.lambda * RATE_WEIGHT * db) / es) as i64;
                }
                if best.is_none_or(|b| cost < b.0) {
                    best = Some((cost, i, delta));
                }
            }
        }
        if let Some((_, i, delta)) = best {
            let l = i32::from(levels[i].unsigned_abs());
            let nl = l + delta;
            nz = nz + u32::from(l == 0) - u32::from(nl == 0);
            levels[i] = if coeffs[i] < 0 { -nl } else { nl } as i16;
        }
    }
    nz
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cabac_enc::CabacEncoder;
    use crate::hevc::residual::{ResidualParams, write_residual};

    fn lcg(s: &mut u64) -> i32 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*s >> 33) as i32
    }

    fn params(b: &Block<'_>) -> ResidualParams {
        ResidualParams {
            log2_size: b.log2,
            c_idx: b.c_idx,
            scan_idx: b.scan_idx,
            bypass: false,
            transform_skip_allowed: false,
            sign_hiding: b.sign_hiding,
            intra: false,
            pred_mode_intra: 0,
            ts_context: false,
            implicit_rdpcm: false,
            explicit_rdpcm: false,
            persistent_rice: false,
            trace: false,
        }
    }

    /// The optimised levels cost no more, in the measure they optimise, than
    /// the plain rounded ones, on random blocks of every size and scan;
    /// every sub-block the hiding rule applies to has the parity it needs
    /// (the writer asserts it while spelling the block).
    #[test]
    fn rdoq_beats_rounding_and_hides_correctly() {
        let mut seed = 99u64;
        for log2 in 2..=5u32 {
            for scan_idx in 0..3u32 {
                if scan_idx != 0 && log2 > 3 {
                    continue;
                }
                for qp in [22, 32, 40] {
                    for sdh in [false, true] {
                        let n = 1usize << log2;
                        let cx = Contexts::new(1, qp);
                        let b = Block {
                            log2,
                            c_idx: 0,
                            scan_idx,
                            qp,
                            bit_depth: 8,
                            lambda: 0.85 * (f64::from(qp - 12) / 3.0).exp2(),
                            cx: &cx,
                            sign_hiding: sdh,
                            cbf_ctx: None,
                        };
                        let coeffs: Vec<i16> = (0..n * n)
                            .map(|i| {
                                let fall = 1 + (i % n + i / n) as i32;
                                ((lcg(&mut seed) % 4000) - 2000) as i16 / fall as i16
                            })
                            .collect();
                        let mut lv = vec![0i16; n * n];
                        let nz = quantise(&b, &coeffs, &mut lv);
                        assert_eq!(nz as usize, lv.iter().filter(|&&v| v != 0).count());
                        // The plain quantiser at half a step, hidden too.
                        let scale = i64::from(crate::dsp::hevc_enc::quant_scale((qp % 6) as usize));
                        let qb = crate::dsp::hevc_enc::qbits(qp, log2, 8);
                        let mut rd: Vec<i16> = coeffs
                            .iter()
                            .map(|&c| {
                                let m = ((i64::from(c.unsigned_abs()) * scale + (1 << (qb - 1)))
                                    >> qb) as i16;
                                if c < 0 { -m } else { m }
                            })
                            .collect();
                        if sdh {
                            hide_signs(&b, &coeffs, &mut rd);
                        }
                        let j = |levels: &[i16]| -> f64 {
                            let tshift = 15 - 8 - log2 as i32;
                            let es = 1.0 / (scale as f64 * scale as f64)
                                * (-2.0 * f64::from(tshift)).exp2();
                            let d: f64 = coeffs
                                .iter()
                                .zip(levels)
                                .map(|(&c, &l)| {
                                    let e = (i64::from(c.unsigned_abs()) * scale
                                        - i64::from(l.unsigned_abs()) * (1i64 << qb))
                                        as f64;
                                    e * e * es
                                })
                                .sum();
                            let r = if levels.iter().any(|&v| v != 0) {
                                let mut e = CabacEncoder::estimating();
                                let mut c = cx.clone();
                                write_residual(&mut e, &mut c, &params(&b), levels);
                                e.fractional_bits()
                            } else {
                                0.0
                            };
                            d + b.lambda * r
                        };
                        let (jo, jr) = (j(&lv), j(&rd));
                        assert!(
                            jo <= jr * 1.02 + 1.0,
                            "log2 {log2} scan {scan_idx} qp {qp} sdh {sdh}: rdoq {jo:.0} rounding {jr:.0}"
                        );
                    }
                }
            }
        }
    }
}
