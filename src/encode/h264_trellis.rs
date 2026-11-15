//! Rate-distortion optimised quantisation of one transform block for a
//! CABAC stream (a trellis over the block's level choices).
//!
//! A dead-zone quantiser rounds each coefficient on its own. What a level
//! *costs* is not its own business, though: CABAC codes a block as a
//! significance map (one flag per position up to the last significant one,
//! a "last" flag after each significant one) and then the levels in reverse
//! scan order, whose contexts depend on how many ones and how many larger
//! levels came before. So whether to round a coefficient up, down or to
//! zero depends on its neighbours in the scan, and the best joint choice is
//! a shortest path:
//!
//! - positions are visited from the highest-frequency candidate down to
//!   the first;
//! - the path state is what the level contexts need — nothing coded yet,
//!   or the run of ones so far (0..=3+) with no larger level, or the count
//!   of larger levels (1..=4+) — nine states;
//! - each step tries the rounded level, one less, and zero, charging the
//!   squared quantisation error in level units plus `lambda` times the bins
//!   the choice adds, priced from the contexts' live probability states.
//!
//! Distortion in level units: the forward transform and the multipliers
//! make a level one quantiser step (`Qstep`) of the orthonormal transform's
//! coefficient, and an orthonormal transform keeps squared error, so a
//! block's pixel-domain squared error is `Qstep^2` times the sum of the
//! squared level errors — and `lambda_ssd / Qstep^2` is the multiplier in
//! those units.

use crate::cabac_enc::state_bits;
use crate::h264::tables::{ZIGZAG4X4, ZIGZAG8X8};
use crate::h264::tables_gen::{LAST_COEFF_8X8_CTX, SIG_COEFF_8X8_CTX};

/// What a block's trellis needs besides its coefficients: the context
/// states the macroblock starts from, and the multiplier.
pub(crate) struct Trellis {
    /// The CABAC context states (`CabacState::ctx`).
    pub(crate) ctx: [u8; 1024],
    /// `lambda_ssd` of the macroblock.
    pub(crate) lambda: f32,
    /// `QpBdOffset`: the quantisers the block coders pass are primed, the
    /// step that pairs with `lambda` is the unprimed one.
    pub(crate) bd_off: i32,
}

/// Which kind of block, for its contexts (frame-coded, Table 9-34 offsets
/// plus Table 9-40 category offsets).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Block {
    /// A luma 4x4 with its DC (ctxBlockCat 2).
    Luma4,
    /// An Intra_16x16 luma AC block (cat 1): positions 1..=15.
    Luma16Ac,
    /// A chroma AC block (cat 4): positions 1..=15.
    ChromaAc,
    /// A luma 8x8 (cat 5).
    Luma8,
}

impl Block {
    /// `(sig base, last base, abs base, cbf base)`; cbf none for 8x8.
    fn bases(self) -> (usize, usize, usize, Option<usize>) {
        match self {
            Block::Luma4 => (134, 195, 247, Some(93)),
            Block::Luma16Ac => (120, 181, 237, Some(89)),
            Block::ChromaAc => (152, 213, 266, Some(101)),
            Block::Luma8 => (402, 417, 426, None),
        }
    }
}

/// `Qstep` at `qp` (8.5's step table, doubling every six).
fn qstep(qp: i32) -> f32 {
    const B: [f32; 6] = [0.625, 0.6875, 0.8125, 0.875, 1.0, 1.125];
    B[(qp.rem_euclid(6)) as usize] * (qp.div_euclid(6) as f32).exp2()
}

/// The bins of a level's Exp-Golomb escape suffix (`abs >= 15`): the
/// unary prefix's `k` ones and terminator plus `k` suffix bits.
#[inline]
fn escape_bins(abs: u32) -> u32 {
    let mut rem = abs - 15;
    let mut k = 0;
    while rem >= (1 << k) {
        rem -= 1 << k;
        k += 1;
    }
    2 * k + 1
}

/// Path states: 0 nothing coded yet; 1..=4 ones seen 0..=3+ with no larger
/// level; 5..=8 larger levels seen 1..=4+. Node 1 (no ones, something
/// coded) cannot be reached — a first level moves to 2 or 5 — so the
/// search holds the other eight, one vector lane each: lane 0 is node 0,
/// lanes 1..=3 nodes 2..=4, lanes 4..=7 nodes 5..=8.
#[cfg(test)]
const NODES: usize = 9;
const L: usize = 8;
type V = [f32; L];
type U = [u32; L];
const INF: f32 = f32::INFINITY;
/// Every lane coming from itself with a zero level.
const STAY: U = [
    0,
    1 << 24,
    2 << 24,
    3 << 24,
    4 << 24,
    5 << 24,
    6 << 24,
    7 << 24,
];
/// Each lane's context increment for a level's first bin, and for the
/// rest of its prefix ([`first_inc`] and [`rest_inc`] of its node).
const FIRST_INC: [usize; L] = [1, 2, 3, 4, 0, 0, 0, 0];
const REST_INC: [usize; L] = [5, 5, 5, 5, 6, 7, 8, 9];

#[cfg(test)]
fn first_inc(node: usize) -> usize {
    match node {
        0 => 1,
        1..=4 => (node).min(4), // 1 + eq1, eq1 = node - 1
        _ => 0,
    }
}

#[cfg(test)]
fn rest_inc(node: usize) -> usize {
    if node >= 5 { 5 + (node - 4).min(4) } else { 5 }
}

/// Lane-wise `a + b`.
#[inline(always)]
fn add(a: V, b: V) -> V {
    std::array::from_fn(|j| a[j] + b[j])
}

/// The lane-wise better of `best` and `cand`: `cand` (and its code) only
/// where strictly cheaper, so of equal costs the one offered first stays.
#[inline(always)]
fn offer(best: &mut V, from: &mut U, cand: V, code: U) {
    for j in 0..L {
        let b = cand[j] < best[j];
        best[j] = if b { cand[j] } else { best[j] };
        from[j] = if b { code[j] } else { from[j] };
    }
}

/// `from` code: the source lane and the level chosen.
const fn code(lane: u32, lv: u32) -> u32 {
    (lane << 24) | lv
}

/// The node a path moves to on a level of `abs` (the definition
/// [`next_table`] tabulates).
#[cfg(test)]
fn next(node: usize, abs: u32) -> usize {
    if abs == 1 {
        match node {
            0 => 2,
            1..=3 => node + 1,
            _ => node, // 4 saturates; a larger level seen keeps its count
        }
    } else {
        match node {
            0..=4 => 5,
            5..=7 => node + 1,
            _ => 8,
        }
    }
}

impl Trellis {
    #[inline]
    fn bits(&self, ctx: usize, bin: u32) -> f32 {
        state_bits(self.ctx[ctx], bin)
    }

    /// Quantise `coeffs` (raster, `n` = 16 or 64) into `levels` by the
    /// trellis: `mf` and `qbits` the quantiser's, `qp` the block's own
    /// quantiser, primed (for its step). AC blocks leave position 0 at zero
    /// (its DC is coded elsewhere). Returns the nonzero count.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn quant(
        &self,
        kind: Block,
        coeffs: &[i32],
        levels: &mut [i16],
        mf: &[i32],
        qbits: u32,
        qp: i32,
    ) -> u32 {
        // Sized to the block: the per-position tables are the trellis's
        // own working set, and a 4x4 block's are a quarter of an 8x8's.
        if kind == Block::Luma8 {
            self.quant_n::<64>(kind, coeffs, levels, mf, qbits, qp)
        } else {
            self.quant_n::<16>(kind, coeffs, levels, mf, qbits, qp)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn quant_n<const N: usize>(
        &self,
        kind: Block,
        coeffs: &[i32],
        levels: &mut [i16],
        mf: &[i32],
        qbits: u32,
        qp: i32,
    ) -> u32 {
        let (scan, start, count): (&[u8], usize, usize) = match kind {
            Block::Luma8 => (&ZIGZAG8X8[..], 0, 64),
            Block::Luma4 => (&ZIGZAG4X4[..], 0, 16),
            _ => (&ZIGZAG4X4[..], 1, 15),
        };
        for l in levels.iter_mut().take(N) {
            *l = 0;
        }
        // Exact levels and their rounded magnitudes, in scan order.
        let scale = 1.0 / (1u64 << qbits) as f32;
        // In raster order first, where the arithmetic is one vector
        // expression, then taken into the scan.
        let mut xr = [0f32; N];
        for ((o, &c), &m) in xr.iter_mut().zip(&coeffs[..N]).zip(&mf[..N]) {
            *o = (c.unsigned_abs() as f32) * m as f32 * scale;
        }
        let mut x = [0f32; N];
        let mut top = [0u32; N];
        let mut last = usize::MAX;
        for i in 0..count {
            let v = xr[scan[start + i] as usize & (N - 1)];
            x[i] = v;
            let t = (v + 0.5) as u32;
            top[i] = t;
            last = if t != 0 { i } else { last };
        }
        if last == usize::MAX {
            return 0;
        }
        let (sig_base, last_base, abs_base, cbf_base) = kind.bases();
        // lambda in this block's level units: squared error scales with its
        // own step (chroma's differs from luma's).
        let own = qstep(qp - self.bd_off);
        let lam = self.lambda / (own * own);

        // Every bin the block can code, priced once and scaled by lambda:
        // the level bins by context increment, then sig / last per position.
        let mut lb = [[0f32; 2]; 10];
        for (inc, e) in lb.iter_mut().enumerate() {
            *e = [
                lam * self.bits(abs_base + inc, 0),
                lam * self.bits(abs_base + inc, 1),
            ];
        }
        // Per lane: the cost of a level of 1, the fixed part of a larger one
        // (first bin 1 plus the rest-context terminator) and the cost per
        // extra unit, all with the sign; and past the capped prefix (no
        // terminator, the escape bypass) the fixed part.
        let mut one = [0f32; L];
        let mut big0 = [0f32; L];
        let mut big1 = [0f32; L];
        let mut esc = [0f32; L];
        for j in 0..L {
            let (f, r) = (FIRST_INC[j], REST_INC[j]);
            one[j] = lb[f][0] + lam;
            big0[j] = lb[f][1] + lb[r][0] + lam;
            big1[j] = lb[r][1];
            esc[j] = big0[j] - lb[r][0] + 13.0 * big1[j];
        }
        let max_idx = count - 1;

        let mut cost = [INF; L];
        // Everything past `last` is zero and costs its squared error only.
        cost[0] = (last + 1..count).map(|i| x[i] * x[i]).sum();
        // Per position and resulting lane: the lane it came from and the
        // level chosen, packed.
        // Written at every position from `last` down before it is read,
        // so it starts uninitialised: zeroing it was a 2 KB memset a block.
        let mut from = [const { std::mem::MaybeUninit::<U>::uninit() }; N];
        for i in (0..=last).rev() {
            let t = top[i];
            let coded_sig = i < max_idx;
            let xi = x[i];
            let d0 = xi * xi;
            // The significance context (only the last position has none).
            let sc = || {
                if count == 64 {
                    sig_base + SIG_COEFF_8X8_CTX[0][i] as usize
                } else {
                    sig_base + i
                }
            };
            if t == 0 {
                let zs0 = if coded_sig {
                    lam * self.bits(sc(), 0)
                } else {
                    0.0
                };
                // Only zero is on offer: every path stays on its node. As
                // one expression over every lane — a dead node's cost is
                // infinite and stays so, and no path reads its entry.
                let mut zv = [zs0; L];
                zv[0] = 0.0;
                cost = add(add(cost, [d0; L]), zv);
                from[i].write(STAY);
                continue;
            }
            let (zs, os, l0, l1) = if coded_sig {
                let sc = sc();
                let lc = if count == 64 {
                    last_base + LAST_COEFF_8X8_CTX[i] as usize
                } else {
                    last_base + i
                };
                (
                    lam * self.bits(sc, 0),
                    lam * self.bits(sc, 1),
                    lam * self.bits(lc, 0),
                    lam * self.bits(lc, 1),
                )
            } else {
                (0.0, 0.0, 0.0, 0.0)
            };
            // `t >= 1` here: the candidates are `t - 1` (when that is a
            // level) and `t`, and zero while `t <= 2`. Each candidate's
            // cost from every lane is one vector expression; where it
            // moves a path depends only on whether the level is one, so
            // which source wins each destination is a fixed gather. The
            // candidates reach a destination in the order a scatter over
            // the nodes visits them (source ascending; zero, `t - 1`, `t`
            // within one) and a later one wins only when strictly cheaper,
            // so of equal costs the same path survives.
            let mut sig = [os + l0; L];
            sig[0] = os + l1;
            let lvl = |lv: u32| -> V {
                let dd = (xi - lv as f32) * (xi - lv as f32);
                let lc = if lv == 1 {
                    one
                } else if lv < 15 {
                    let k = (lv - 2) as f32;
                    std::array::from_fn(|j| big0[j] + k * big1[j])
                } else {
                    let e = lam * escape_bins(lv) as f32;
                    std::array::from_fn(|j| esc[j] + e)
                };
                add(add(add(cost, [dd; L]), sig), lc)
            };
            let lt = lvl(t);
            let (nc, nf) = if t >= 3 {
                // Two larger levels, no zero: lanes 0..=3 all move to 4,
                // 4..=7 up one (7 saturating).
                let lo = t - 1;
                let ll = lvl(lo);
                let mut nc = [INF, INF, INF, INF, ll[0], ll[4], ll[5], ll[6]];
                let mut nf = [
                    0,
                    0,
                    0,
                    0,
                    code(0, lo),
                    code(4, lo),
                    code(5, lo),
                    code(6, lo),
                ];
                offer(
                    &mut nc,
                    &mut nf,
                    [INF, INF, INF, INF, lt[0], lt[4], lt[5], lt[6]],
                    [0, 0, 0, 0, code(0, t), code(4, t), code(5, t), code(6, t)],
                );
                offer(
                    &mut nc,
                    &mut nf,
                    [INF, INF, INF, INF, ll[1], INF, INF, ll[7]],
                    [0, 0, 0, 0, code(1, lo), 0, 0, code(7, lo)],
                );
                offer(
                    &mut nc,
                    &mut nf,
                    [INF, INF, INF, INF, lt[1], INF, INF, lt[7]],
                    [0, 0, 0, 0, code(1, t), 0, 0, code(7, t)],
                );
                for s in 2..4 {
                    for (c, lv) in [(ll[s], lo), (lt[s], t)] {
                        if c < nc[4] {
                            nc[4] = c;
                            nf[4] = code(s as u32, lv);
                        }
                    }
                }
                (nc, nf)
            } else {
                // Zero stays on its lane.
                let mut zv = [zs; L];
                zv[0] = 0.0;
                let z = add(add(cost, [d0; L]), zv);
                // A one moves lanes 0..=2 up one, 3 and 4..=7 stay.
                let l1v = if t == 1 { lt } else { lvl(1) };
                let mut nc = [z[0], l1v[0], l1v[1], l1v[2], z[4], z[5], z[6], z[7]];
                let mut nf = [
                    code(0, 0),
                    code(0, 1),
                    code(1, 1),
                    code(2, 1),
                    code(4, 0),
                    code(5, 0),
                    code(6, 0),
                    code(7, 0),
                ];
                if t == 1 {
                    offer(
                        &mut nc,
                        &mut nf,
                        [INF, z[1], z[2], z[3], l1v[4], l1v[5], l1v[6], l1v[7]],
                        [
                            0,
                            code(1, 0),
                            code(2, 0),
                            code(3, 0),
                            code(4, 1),
                            code(5, 1),
                            code(6, 1),
                            code(7, 1),
                        ],
                    );
                    offer(
                        &mut nc,
                        &mut nf,
                        [INF, INF, INF, l1v[3], INF, INF, INF, INF],
                        [0, 0, 0, code(3, 1), 0, 0, 0, 0],
                    );
                } else {
                    // ... and a two moves lanes 0..=3 to 4, 4..=7 up one.
                    // Lane 4's candidates in order: the twos from lanes 0
                    // to 3, then lane 4's zero and one.
                    nc = [z[0], l1v[0], l1v[1], l1v[2], lt[0], lt[4], lt[5], lt[6]];
                    nf = [
                        code(0, 0),
                        code(0, 1),
                        code(1, 1),
                        code(2, 1),
                        code(0, 2),
                        code(4, 2),
                        code(5, 2),
                        code(6, 2),
                    ];
                    offer(
                        &mut nc,
                        &mut nf,
                        [INF, z[1], z[2], z[3], lt[1], z[5], z[6], z[7]],
                        [
                            0,
                            code(1, 0),
                            code(2, 0),
                            code(3, 0),
                            code(1, 2),
                            code(5, 0),
                            code(6, 0),
                            code(7, 0),
                        ],
                    );
                    offer(
                        &mut nc,
                        &mut nf,
                        [INF, INF, INF, l1v[3], lt[2], l1v[5], l1v[6], l1v[7]],
                        [
                            0,
                            0,
                            0,
                            code(3, 1),
                            code(2, 2),
                            code(5, 1),
                            code(6, 1),
                            code(7, 1),
                        ],
                    );
                    offer(
                        &mut nc,
                        &mut nf,
                        [INF, INF, INF, INF, lt[3], INF, INF, lt[7]],
                        [0, 0, 0, 0, code(3, 2), 0, 0, code(7, 2)],
                    );
                    for (c, cd) in [(z[4], code(4, 0)), (l1v[4], code(4, 1))] {
                        if c < nc[4] {
                            nc[4] = c;
                            nf[4] = cd;
                        }
                    }
                }
                (nc, nf)
            };
            cost = nc;
            from[i].write(nf);
        }
        // The coded block flag: an all-zero block saves it.
        if let Some(cb) = cbf_base {
            let c = cb + 1;
            cost[0] += lam * self.bits(c, 0);
            let one = lam * self.bits(c, 1);
            for v in cost.iter_mut().skip(1) {
                *v += one;
            }
        }
        let mut lane = 0;
        for j in 1..L {
            if cost[j] < cost[lane] {
                lane = j;
            }
        }
        let mut nz = 0;
        for i in 0..=last {
            // SAFETY: every position up to `last` was written above.
            let e = unsafe { from[i].assume_init_ref() }[lane];
            let lv = e & 0xff_ffff;
            if lv != 0 {
                let r = scan[start + i] as usize;
                levels[r] = if coeffs[r] < 0 {
                    -(lv as i16)
                } else {
                    lv as i16
                };
                nz += 1;
            }
            lane = (e >> 24) as usize;
        }
        nz
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Timing of the trellis on blocks shaped like a coded picture's
    /// (`cargo test --release -- --ignored trellis_bench --nocapture`),
    /// with a checksum of what it chose.
    #[test]
    #[ignore]
    fn trellis_bench() {
        use crate::dsp::h264_enc::{Quant, qbits4, qbits8};
        use crate::h264::sps::ScalingLists;
        let lists = ScalingLists {
            list4x4: [[16; 16]; 6],
            list8x8: [[16; 64]; 6],
        };
        let q = Quant::new(&lists);
        let mut ctx = [0u8; 1024];
        crate::h264::cabac_mb::init_contexts(&mut ctx, crate::h264::SliceType::P, 0, 27);
        let t = Trellis {
            ctx,
            lambda: crate::encode::h264_analyse::lambda_ssd(27),
            bd_off: 0,
        };
        let mut seed = 1u64;
        let mut rnd = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as u32
        };
        // Laplacian-ish coefficients decaying with frequency.
        let mut blocks8 = Vec::new();
        let mut blocks4 = Vec::new();
        for _ in 0..4000 {
            let mut b = [0i32; 64];
            for (i, v) in b.iter_mut().enumerate() {
                let (x, y) = (i % 8, i / 8);
                let scale = 2000.0 / (1.0 + (x + y) as f32 * 1.5);
                let u = (rnd() as f32 + 1.0) / 4294967296.0 * 2.0;
                let m = -(u.min(0.999)).ln() * scale * 0.3;
                *v = if rnd() & 1 == 0 {
                    m as i32
                } else {
                    -(m as i32)
                };
            }
            blocks8.push(b);
            let mut c = [0i32; 16];
            for (i, v) in c.iter_mut().enumerate() {
                *v = b[(i / 4) * 8 + i % 4] / 4;
            }
            blocks4.push(c);
        }
        if let Some(path) = std::env::var_os("TRBENCH") {
            let d = std::fs::read(path).unwrap();
            let mut recs = Vec::new();
            let mut o = 0;
            while o < d.len() {
                let (kind, qbits, qp, k) =
                    (d[o], d[o + 1] as u32, d[o + 2] as i32, d[o + 3] as usize);
                let lambda = f32::from_le_bytes(d[o + 4..o + 8].try_into().unwrap());
                o += 8;
                let rd = |o: usize| i32::from_le_bytes(d[o..o + 4].try_into().unwrap());
                let c: Vec<i32> = (0..k).map(|i| rd(o + 4 * i)).collect();
                let m: Vec<i32> = (0..k).map(|i| rd(o + 4 * k + 4 * i)).collect();
                o += 8 * k;
                let kind =
                    [Block::Luma4, Block::Luma16Ac, Block::ChromaAc, Block::Luma8][kind as usize];
                recs.push((kind, qbits, qp, lambda, c, m));
            }
            let mut tt = Trellis {
                ctx: t.ctx,
                lambda: 0.0,
                bd_off: 0,
            };
            let mut sum = 0u64;
            let start = std::time::Instant::now();
            recs.truncate(20000);
            for _ in 0..75 {
                for (kind, qbits, qp, lambda, c, m) in &recs {
                    tt.lambda = *lambda;
                    let mut l = [0i16; 64];
                    sum += tt.quant(*kind, c, &mut l, m, *qbits, *qp) as u64;
                    sum += l.iter().map(|&v| v.unsigned_abs() as u64).sum::<u64>();
                }
            }
            let el = start.elapsed();
            eprintln!(
                "trellis dump bench: {} blocks, {:.1} ns per block, checksum {sum}",
                recs.len(),
                el.as_nanos() as f64 / (75 * recs.len()) as f64
            );
            return;
        }
        let mut sum = 0u64;
        let start = std::time::Instant::now();
        for _ in 0..50 {
            for (b8, b4) in blocks8.iter().zip(&blocks4) {
                let mut l8 = [0i16; 64];
                sum += t.quant(Block::Luma8, b8, &mut l8, &q.mf8[1][27 % 6], qbits8(27), 27) as u64;
                sum += l8.iter().map(|&v| v.unsigned_abs() as u64).sum::<u64>();
                let mut l4 = [0i16; 16];
                sum += t.quant(Block::Luma4, b4, &mut l4, &q.mf4[3][27 % 6], qbits4(27), 27) as u64;
                sum += l4.iter().map(|&v| v.unsigned_abs() as u64).sum::<u64>();
            }
        }
        let el = start.elapsed();
        eprintln!(
            "trellis bench: {:.1} ns per 8x8+4x4 pair, checksum {sum}",
            el.as_nanos() as f64 / 200000.0
        );
    }

    /// The lane tables and the fixed gathers agree with the node
    /// definitions they were derived from.
    #[test]
    fn lanes_are_nodes() {
        const NODE: [usize; L] = [0, 2, 3, 4, 5, 6, 7, 8];
        for j in 0..L {
            assert_eq!(FIRST_INC[j], first_inc(NODE[j]));
            assert_eq!(REST_INC[j], rest_inc(NODE[j]));
        }
        for n in 0..NODES {
            for abs in 1..4 {
                assert_ne!(next(n, abs), 1, "node 1 is unreachable");
            }
        }
        // ONE moves lanes 0..=3 up one (3 saturates), BIG moves 0..=3 to 4
        // and 4..=7 up one (7 saturates): what the gathers in `quant_n` spell.
        for j in 0..L {
            let one = NODE.iter().position(|&n| n == next(NODE[j], 1)).unwrap();
            let big = NODE.iter().position(|&n| n == next(NODE[j], 2)).unwrap();
            assert_eq!(one, if j < 3 { j + 1 } else { j });
            assert_eq!(big, if j < 4 { 4 } else { (j + 1).min(7) });
        }
    }
}
