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
    /// Run the general path's lanes as vectors where the CPU has them
    /// ([`Trellis::simd_default`]); the levels are the scalar path's.
    /// Only AArch64 has the vector path, so elsewhere it is set, never read.
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    pub(crate) simd: bool,
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

/// Below this exact level (in quantiser steps) every coefficient of a
/// block is zero without a trellis: where the plain inter quantiser's
/// rounding offset of 1/6 leaves a level of zero.
const DEAD_ZONE: f32 = 5.0 / 6.0;

/// `Qstep` at `qp` (8.5's step table, doubling every six). The doubling
/// is the power of two built from its exponent bits — exactly what `exp2`
/// returns at an integer, without the library call every block paid.
fn qstep(qp: i32) -> f32 {
    const B: [f32; 6] = [0.625, 0.6875, 0.8125, 0.875, 1.0, 1.125];
    B[(qp.rem_euclid(6)) as usize] * pow2(qp.div_euclid(6))
}

/// `2^e` for a normal exponent.
#[inline]
fn pow2(e: i32) -> f32 {
    debug_assert!((-126..=127).contains(&e));
    f32::from_bits(((e + 127) as u32) << 23)
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

/// A block's level prices per lane (see `quant_n`).
struct LevelCosts {
    one: V,
    big0: V,
    big1: V,
    esc: V,
    lam: f32,
}

/// Every lane's cost after coding a level of `lv` at a position whose
/// exact level is `xi`: `cost + dd + sig + level bins`, in that order.
#[inline(always)]
fn level(c: &LevelCosts, cost: V, xi: f32, sig: V, lv: u32) -> V {
    let dd = (xi - lv as f32) * (xi - lv as f32);
    let lc = if lv == 1 {
        c.one
    } else if lv < 15 {
        let k = (lv - 2) as f32;
        std::array::from_fn(|j| c.big0[j] + k * c.big1[j])
    } else {
        escape_cost(c, lv)
    };
    add(add(add(cost, [dd; L]), sig), lc)
}

/// A level past the capped prefix: the fixed part and the escape bins.
#[cold]
#[inline(never)]
fn escape_cost(c: &LevelCosts, lv: u32) -> V {
    let e = c.lam * escape_bins(lv) as f32;
    std::array::from_fn(|j| c.esc[j] + e)
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
    /// (its DC is coded elsewhere). `levels` must arrive zeroed; only the
    /// nonzero positions are written. Returns the nonzero count.
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
        // `levels` arrives zeroed (every caller's is a fresh array): a
        // second zeroing here was a call to `bzero` per 8x8 block, most of
        // which then quantise to nothing.
        debug_assert!(levels[..N].iter().all(|&l| l == 0), "levels arrive zeroed");
        // Exact levels and their rounded magnitudes, in scan order.
        let scale = 1.0 / (1u64 << qbits) as f32;
        // In raster order first, where the arithmetic is one vector
        // expression, then taken into the scan.
        // Built element by element: a zeroed array first was a call to
        // `memset` per 8x8 block.
        let (coeffs, mf) = (&coeffs[..N], &mf[..N]);
        #[cfg(target_arch = "aarch64")]
        let (x, top, last, top_max) = if self.simd {
            // SAFETY: NEON is baseline on AArch64; the lengths are `N`.
            match unsafe { prepass_neon::<N>(coeffs, mf, scale, start) } {
                Some(p) => p,
                None => return 0,
            }
        } else {
            match prepass::<N>(coeffs, mf, scale, scan, start, count) {
                Some(p) => p,
                None => return 0,
            }
        };
        #[cfg(not(target_arch = "aarch64"))]
        let Some((x, top, last, top_max)) = prepass::<N>(coeffs, mf, scale, scan, start, count)
        else {
            return 0;
        };
        let (sig_base, last_base, abs_base, cbf_base) = kind.bases();
        // lambda in this block's level units: squared error scales with its
        // own step (chroma's differs from luma's).
        let own = qstep(qp - self.bd_off);
        let lam = self.lambda / (own * own);
        if top_max == 1 {
            return self.quant_ones::<N>(kind, coeffs, levels, &x, &top, last, lam);
        }

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
        let costs = LevelCosts {
            one,
            big0,
            big1,
            esc,
            lam,
        };
        let max_idx = count - 1;
        #[cfg(target_arch = "aarch64")]
        if self.simd {
            // SAFETY: NEON is baseline on AArch64.
            return unsafe { self.general_neon::<N>(kind, coeffs, levels, &x, &top, last, &costs) };
        }

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
            let lvl = |lv: u32| -> V { level(&costs, cost, xi, sig, lv) };
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
            nz += put_level(levels, coeffs, scan[start + i] as usize, e & 0xff_ffff);
            lane = (e >> 24) as usize;
        }
        nz
    }
}

/// [`Trellis::quant_n`]'s first pass, from a block's coefficients in
/// raster order: each position's exact level in scan order (`x`, past
/// the scan zero), its rounded magnitude (`top`), the last position
/// rounding to a level, and the largest rounded level — or `None` for a
/// block the trellis leaves all zero.
#[inline(always)]
fn prepass<const N: usize>(
    coeffs: &[i32],
    mf: &[i32],
    scale: f32,
    scan: &[u8],
    start: usize,
    count: usize,
) -> Option<([f32; N], [u32; N], usize, u32)> {
    // In raster order first, where the arithmetic is one vector
    // expression, then taken into the scan.
    // Built element by element: a zeroed array first was a call to
    // `memset` per 8x8 block.
    let xr: [f32; N] =
        std::array::from_fn(|i| (coeffs[i].unsigned_abs() as f32) * mf[i] as f32 * scale);
    // A block the plain inter quantiser (a dead zone rounding at 1/6)
    // leaves all zero is not trellised: it codes nothing. Of such blocks
    // the trellis itself kept a level in 2 % (park_joy, veryfast and
    // medium), and leaving them zero measured better at medium and
    // slow and next to no worse at veryfast (BD-rate Y against x264:
    // +0.39 -> +0.23 %, +1.38 -> +1.25 %, -2.80 -> -2.66 %) for 3 %
    // fewer instructions. Most of
    // the blocks are below a half (44 % of the calls) and cost nothing
    // either way; the rest (a fifth of the calls) were a whole trellis
    // pass each. Found from the largest exact level before the scan is
    // gathered: the magnitudes are non-negative, so their bit patterns
    // order as they do (an integer maximum, one vector reduction). An
    // AC block's raster position 0 is outside its scan.
    let mx = xr[start..].iter().map(|v| v.to_bits()).max().unwrap_or(0);
    if f32::from_bits(mx) < DEAD_ZONE {
        return None;
    }
    // Built element by element (no zeroing pass first: for an 8x8 that
    // was two calls into `memset`); an AC block's sixteenth is unused.
    let x: [f32; N] = std::array::from_fn(|i| {
        if i < count {
            xr[scan[start + i] as usize & (N - 1)]
        } else {
            0.0
        }
    });
    let top: [u32; N] = std::array::from_fn(|i| (x[i] + 0.5) as u32);
    let mut last = usize::MAX;
    let mut top_max = 0;
    for (i, &t) in top.iter().enumerate().take(count) {
        top_max = top_max.max(t);
        last = if t != 0 { i } else { last };
    }
    if last == usize::MAX {
        return None;
    }
    Some((x, top, last, top_max))
}

/// [`prepass`]'s scan gathers as byte lookups: for each scan position's
/// four bytes, where they sit among a block's raster `f32`s (0xff past the
/// scan, which reads as zero). The 4x4 scan from position 0 and from 1
/// (the AC blocks'), each four vectors.
#[cfg(target_arch = "aarch64")]
static GATHER4: [[[u8; 16]; 4]; 2] = {
    let mut t = [[[0xffu8; 16]; 4]; 2];
    let mut st = 0;
    while st < 2 {
        let mut i = 0;
        while i + st < 16 {
            let r = ZIGZAG4X4[st + i] as usize;
            let mut b = 0;
            while b < 4 {
                t[st][i / 4][(i % 4) * 4 + b] = (4 * r + b) as u8;
                b += 1;
            }
            i += 1;
        }
        st += 1;
    }
    t
};

/// The 8x8 scan's: sixteen vectors, each looked up in the four 64-byte
/// quarters of the block in turn (an index outside a quarter is out of
/// range there, and leaves what an earlier quarter found).
#[cfg(target_arch = "aarch64")]
static GATHER8: [[[u8; 16]; 4]; 16] = {
    let mut t = [[[0xffu8; 16]; 4]; 16];
    let mut i = 0;
    while i < 64 {
        let r = ZIGZAG8X8[i] as usize;
        let mut b = 0;
        while b < 4 {
            let byte = 4 * r + b;
            t[i / 4][byte / 64][(i % 4) * 4 + b] = (byte % 64) as u8;
            b += 1;
        }
        i += 1;
    }
    t
};

/// [`prepass`] in NEON: the same values (each lane's arithmetic is the
/// scalar one's, and the conversions round alike), the scan gathered by
/// table lookups, and the last position and the largest level found from
/// a mask and a reduction rather than a pass over the positions.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn prepass_neon<const N: usize>(
    coeffs: &[i32],
    mf: &[i32],
    scale: f32,
    start: usize,
) -> Option<([f32; N], [u32; N], usize, u32)> {
    use std::arch::aarch64::*;
    debug_assert!(coeffs.len() >= N && mf.len() >= N && (N == 16 || N == 64));
    unsafe {
        // Every array below is written whole before it is read, so none
        // is zeroed first (for an 8x8 a zeroing is a call to `memset`).
        let mut xr = std::mem::MaybeUninit::<[f32; N]>::uninit();
        let xp = xr.as_mut_ptr() as *mut f32;
        let mut mx = vdupq_n_u32(0);
        for k in 0..N / 4 {
            let c = vreinterpretq_u32_s32(vabsq_s32(vld1q_s32(coeffs.as_ptr().add(4 * k))));
            let m = vcvtq_f32_s32(vld1q_s32(mf.as_ptr().add(4 * k)));
            let v = vmulq_n_f32(vmulq_f32(vcvtq_f32_u32(c), m), scale);
            vst1q_f32(xp.add(4 * k), v);
            mx = vmaxq_u32(mx, vreinterpretq_u32_f32(v));
        }
        if start != 0 {
            // An AC block's raster position 0 is outside its scan.
            let v = vreinterpretq_u32_f32(vld1q_f32(xp));
            let mut m = vsetq_lane_u32::<0>(0, v);
            for k in 1..N / 4 {
                m = vmaxq_u32(m, vreinterpretq_u32_f32(vld1q_f32(xp.add(4 * k))));
            }
            mx = m;
        }
        if f32::from_bits(vmaxvq_u32(mx)) < DEAD_ZONE {
            return None;
        }
        let p = xp as *const u8;
        let mut x = std::mem::MaybeUninit::<[f32; N]>::uninit();
        let mut top = std::mem::MaybeUninit::<[u32; N]>::uninit();
        let (xo, to) = (x.as_mut_ptr() as *mut f32, top.as_mut_ptr() as *mut u32);
        let half = vdupq_n_f32(0.5);
        let mut tmax = vdupq_n_u32(0);
        let mut nz = std::mem::MaybeUninit::<[u8; N]>::uninit();
        let nzp = nz.as_mut_ptr() as *mut u8;
        let gather = |k: usize| -> uint8x16_t {
            if N == 16 {
                let g = &GATHER4[start];
                vqtbl4q_u8(vld1q_u8_x4(p), vld1q_u8(g[k].as_ptr()))
            } else {
                let g = &GATHER8[k];
                let mut v = vqtbl4q_u8(vld1q_u8_x4(p), vld1q_u8(g[0].as_ptr()));
                v = vqtbx4q_u8(v, vld1q_u8_x4(p.add(64)), vld1q_u8(g[1].as_ptr()));
                v = vqtbx4q_u8(v, vld1q_u8_x4(p.add(128)), vld1q_u8(g[2].as_ptr()));
                vqtbx4q_u8(v, vld1q_u8_x4(p.add(192)), vld1q_u8(g[3].as_ptr()))
            }
        };
        for k in 0..N / 4 {
            let v = vreinterpretq_f32_u8(gather(k));
            vst1q_f32(xo.add(4 * k), v);
            let t = vcvtq_u32_f32(vaddq_f32(v, half));
            vst1q_u32(to.add(4 * k), t);
            tmax = vmaxq_u32(tmax, t);
            // Whether each rounds to a level, a byte per position.
            let n = vmovn_u16(vcombine_u16(vmovn_u32(vtstq_u32(t, t)), vdup_n_u16(0)));
            vst1_lane_u32::<0>(nzp.add(4 * k) as *mut u32, vreinterpret_u32_u8(n));
        }
        // The positions' bytes to bits: the last one set is `last`.
        const W: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        let w = vld1q_u8(W.as_ptr());
        let mut mask = 0u64;
        for q in 0..N / 16 {
            let m = vandq_u8(vld1q_u8(nzp.add(16 * q)), w);
            let bits =
                (vaddv_u8(vget_low_u8(m)) as u64) | ((vaddv_u8(vget_high_u8(m)) as u64) << 8);
            mask |= bits << (16 * q);
        }
        // A block past the dead zone has a level that rounds to one.
        debug_assert!(mask != 0);
        if mask == 0 {
            return None;
        }
        Some((
            x.assume_init(),
            top.assume_init(),
            63 - mask.leading_zeros() as usize,
            vmaxvq_u32(tmax),
        ))
    }
}

/// Write the level `lv` chosen at raster position `r` with its
/// coefficient's sign, a zero included (`levels` arrives zeroed and the
/// walk back writes each position once), and count it when it is not
/// zero — without a branch on the level: which positions keep one is as
/// good as random, and the walk mispredicted about one in two of them
/// (3 to 4 % of every tier's cycles).
#[inline(always)]
fn put_level(levels: &mut [i16], coeffs: &[i32], r: usize, lv: u32) -> u32 {
    let v = lv as i16;
    levels[r] = if coeffs[r] < 0 { -v } else { v };
    (lv != 0) as u32
}

impl Trellis {
    /// Whether [`Trellis::simd`] should be on: where the `h264_enc` table
    /// would take its NEON kernels — an AArch64 CPU with NEON, not capped
    /// by `H26X_NO_SIMD` / `H26X_MAX_SIMD`, and not kept scalar by
    /// `H26X_ENC_NO_SIMD`.
    pub(crate) fn simd_default() -> bool {
        cfg!(target_arch = "aarch64")
            && crate::dsp::Cpu::detect_honouring_env().neon
            && !crate::dsp::enc_simd_disabled("h264_enc")
    }
}

#[cfg(target_arch = "aarch64")]
impl Trellis {
    /// [`Trellis::quant_n`]'s general path with each half of the eight
    /// lanes (nodes 0 and 2..=4, then 5..=8) in one NEON vector. Every
    /// lane's arithmetic is the scalar path's, operation for operation —
    /// the same additions in the same order, the products not fused — and
    /// every destination lane takes its candidates in the scalar path's
    /// order, a later one only where strictly cheaper, so the costs, the
    /// paths and the levels are the same bits.
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "neon")]
    unsafe fn general_neon<const N: usize>(
        &self,
        kind: Block,
        coeffs: &[i32],
        levels: &mut [i16],
        x: &[f32; N],
        top: &[u32; N],
        last: usize,
        c: &LevelCosts,
    ) -> u32 {
        use std::arch::aarch64::*;
        type F = float32x4_t;
        type C = uint32x4_t;
        #[inline(always)]
        unsafe fn ld(a: &[f32]) -> F {
            unsafe { vld1q_f32(a.as_ptr()) }
        }
        #[inline(always)]
        unsafe fn ldu(a: [u32; 4]) -> C {
            unsafe { vld1q_u32(a.as_ptr()) }
        }
        /// Per lane, `b` (and its code) where strictly cheaper than `a`,
        /// else `a`: the first minimum of the two in that order. Taking a
        /// run of candidates one at a time, a later one only where strictly
        /// cheaper, leaves the first minimum of the run — and the first
        /// minimum of a run is the first minimum of its two halves' first
        /// minima, so a run can be reduced as a tree, shorter than the
        /// chain, with the same winner and the same bits.
        #[inline(always)]
        unsafe fn fmin(a: F, ac: C, b: F, bc: C) -> (F, C) {
            unsafe {
                let m = vcltq_f32(b, a);
                (vbslq_f32(m, b, a), vbslq_u32(m, bc, ac))
            }
        }
        /// `[a0, b0, b1, b2]`.
        #[inline(always)]
        unsafe fn shift_in(a: F, b: F) -> F {
            unsafe { vextq_f32::<3>(vdupq_laneq_f32::<0>(a), b) }
        }
        /// `(lane << 24) | lv` per lane.
        #[inline(always)]
        unsafe fn codes(lanes: [u32; 4], lv: u32) -> C {
            unsafe { vorrq_u32(ldu(lanes.map(|l| l << 24)), vdupq_n_u32(lv)) }
        }

        let (scan, start, count): (&[u8], usize, usize) = match kind {
            Block::Luma8 => (&ZIGZAG8X8[..], 0, 64),
            Block::Luma4 => (&ZIGZAG4X4[..], 0, 16),
            _ => (&ZIGZAG4X4[..], 1, 15),
        };
        let (sig_base, last_base, _, cbf_base) = kind.bases();
        let lam = c.lam;
        let max_idx = count - 1;
        unsafe {
            let inf = vdupq_n_f32(INF);
            let (one_lo, one_hi) = (ld(&c.one[..4]), ld(&c.one[4..]));
            let (big0_lo, big0_hi) = (ld(&c.big0[..4]), ld(&c.big0[4..]));
            let (big1_lo, big1_hi) = (ld(&c.big1[..4]), ld(&c.big1[4..]));
            let (esc_lo, esc_hi) = (ld(&c.esc[..4]), ld(&c.esc[4..]));
            let stay_lo = ldu([0, 1 << 24, 2 << 24, 3 << 24]);
            let stay_hi = ldu([4 << 24, 5 << 24, 6 << 24, 7 << 24]);

            let mut c_lo = inf;
            let c_hi = inf;
            // Everything past `last` is zero and costs its squared error.
            let tail: f32 = (last + 1..count).map(|i| x[i] * x[i]).sum();
            c_lo = vsetq_lane_f32::<0>(tail, c_lo);
            let mut c_hi = c_hi;
            let mut from = [const { std::mem::MaybeUninit::<U>::uninit() }; N];
            // Each position's source lanes again, times eight, a byte each:
            // the walk back's chain (see `quant_ones`).
            let mut src = [const { std::mem::MaybeUninit::<u64>::uninit() }; N];
            let src_bytes = |lo: C, hi: C| -> u64 {
                let lo = vandq_u32(vshrq_n_u32::<21>(lo), vdupq_n_u32(0x38));
                let hi = vandq_u32(vshrq_n_u32::<21>(hi), vdupq_n_u32(0x38));
                let b = vmovn_u16(vcombine_u16(vmovn_u32(lo), vmovn_u32(hi)));
                vget_lane_u64::<0>(vreinterpret_u64_u8(b))
            };
            for i in (0..=last).rev() {
                let t = top[i];
                let coded_sig = i < max_idx;
                let xi = x[i];
                let d0 = xi * xi;
                let dv = vdupq_n_f32(d0);
                let sc = || {
                    if count == 64 {
                        sig_base + SIG_COEFF_8X8_CTX[0][i] as usize
                    } else {
                        sig_base + i
                    }
                };
                let fp = from[i].as_mut_ptr() as *mut u32;
                if t == 0 {
                    let zs0 = if coded_sig {
                        lam * self.bits(sc(), 0)
                    } else {
                        0.0
                    };
                    let z = vdupq_n_f32(zs0);
                    c_lo = vaddq_f32(vaddq_f32(c_lo, dv), vsetq_lane_f32::<0>(0.0, z));
                    c_hi = vaddq_f32(vaddq_f32(c_hi, dv), z);
                    vst1q_u32(fp, stay_lo);
                    vst1q_u32(fp.add(4), stay_hi);
                    src[i].write(0x3830_2820_1810_0800);
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
                let sig_hi = vdupq_n_f32(os + l0);
                let sig_lo = vsetq_lane_f32::<0>(os + l1, sig_hi);
                // `level`: cost + dd + sig + the level's bins, per lane.
                let lvl = |lv: u32| -> (F, F) {
                    let dd = vdupq_n_f32((xi - lv as f32) * (xi - lv as f32));
                    let (lc_lo, lc_hi) = if lv == 1 {
                        (one_lo, one_hi)
                    } else if lv < 15 {
                        let k = (lv - 2) as f32;
                        (
                            vaddq_f32(big0_lo, vmulq_n_f32(big1_lo, k)),
                            vaddq_f32(big0_hi, vmulq_n_f32(big1_hi, k)),
                        )
                    } else {
                        let e = vdupq_n_f32(lam * escape_bins(lv) as f32);
                        (vaddq_f32(esc_lo, e), vaddq_f32(esc_hi, e))
                    };
                    (
                        vaddq_f32(vaddq_f32(vaddq_f32(c_lo, dd), sig_lo), lc_lo),
                        vaddq_f32(vaddq_f32(vaddq_f32(c_hi, dd), sig_hi), lc_hi),
                    )
                };
                let (lt_lo, lt_hi) = lvl(t);
                let (nc_lo, nf_lo, nc_hi, nf_hi);
                if t >= 3 {
                    // Lanes 0..=3 all move to 4, 4..=7 up one (7
                    // saturating); no zero. Each destination's candidates
                    // in the scalar order — lane 4: lane 0's `t - 1` then
                    // `t`, lane 1's, 2's, 3's; lanes 5 and 6: lane 4's and
                    // 5's pair; lane 7: 6's pair then 7's — taken as a
                    // tree of first minima.
                    let lo = t - 1;
                    let (ll_lo, ll_hi) = lvl(lo);
                    let (m_lo, mf_lo) = fmin(
                        ll_lo,
                        codes([0, 1, 2, 3], lo),
                        lt_lo,
                        codes([0, 1, 2, 3], t),
                    );
                    let (m_hi, mf_hi) = fmin(
                        ll_hi,
                        codes([4, 5, 6, 7], lo),
                        lt_hi,
                        codes([4, 5, 6, 7], t),
                    );
                    // Lanes 0 and 2: (0, 1) and (2, 3); then lane 0: all.
                    let (p, pf) = fmin(m_lo, mf_lo, vrev64q_f32(m_lo), vrev64q_u32(mf_lo));
                    let (r, rf) = fmin(p, pf, vextq_f32::<2>(p, p), vextq_u32::<2>(pf, pf));
                    // Lane 2: (6's pair, 7's pair).
                    let (q, qf) = fmin(m_hi, mf_hi, vrev64q_f32(m_hi), vrev64q_u32(mf_hi));
                    let h = vcopyq_laneq_f32::<3, 2>(shift_in(r, m_hi), q);
                    let hf = vcopyq_laneq_u32::<3, 2>(
                        vextq_u32::<3>(vdupq_laneq_u32::<0>(rf), mf_hi),
                        qf,
                    );
                    (nc_lo, nf_lo, nc_hi, nf_hi) = (inf, vdupq_n_u32(0), h, hf);
                } else {
                    // Zero stays on its lane.
                    let z_lo = vaddq_f32(
                        vaddq_f32(c_lo, dv),
                        vsetq_lane_f32::<0>(0.0, vdupq_n_f32(zs)),
                    );
                    let z_hi = vaddq_f32(vaddq_f32(c_hi, dv), vdupq_n_f32(zs));
                    let (l1_lo, l1_hi) = if t == 1 { (lt_lo, lt_hi) } else { lvl(1) };
                    // Lanes 0..=3: lane 0 keeps its zero, a one moves 0..=2
                    // up one; then 1..=3's zeros, then lane 3's one.
                    let (l, lf) = fmin(
                        shift_in(z_lo, l1_lo),
                        ldu([0, 1, (1 << 24) | 1, (2 << 24) | 1]),
                        vsetq_lane_f32::<0>(INF, z_lo),
                        ldu([0, 1 << 24, 2 << 24, 3 << 24]),
                    );
                    let (l, lf) = fmin(
                        l,
                        lf,
                        vcopyq_laneq_f32::<3, 3>(inf, l1_lo),
                        ldu([0, 0, 0, (3 << 24) | 1]),
                    );
                    let (h, hf) = if t == 1 {
                        // Lanes 4..=7: zero, then a one, each staying.
                        fmin(z_hi, stay_hi, l1_hi, codes([4, 5, 6, 7], 1))
                    } else {
                        // A two moves lanes 0..=3 to 4, 4..=7 up one;
                        // lane 4's candidates: the twos from lanes 0..=3,
                        // then its own zero and one.
                        let (a, af) = fmin(
                            shift_in(lt_lo, lt_hi),
                            codes([0, 4, 5, 6], 2),
                            vcopyq_laneq_f32::<0, 1>(z_hi, lt_lo),
                            ldu([(1 << 24) | 2, 5 << 24, 6 << 24, 7 << 24]),
                        );
                        let (b, bf) = fmin(
                            vcopyq_laneq_f32::<0, 2>(l1_hi, lt_lo),
                            ldu([(2 << 24) | 2, (5 << 24) | 1, (6 << 24) | 1, (7 << 24) | 1]),
                            vcopyq_laneq_f32::<3, 3>(vcopyq_laneq_f32::<0, 3>(inf, lt_lo), lt_hi),
                            ldu([(3 << 24) | 2, 0, 0, (7 << 24) | 2]),
                        );
                        let (e, ef) = fmin(
                            vcopyq_laneq_f32::<0, 0>(inf, z_hi),
                            ldu([4 << 24, 0, 0, 0]),
                            vcopyq_laneq_f32::<0, 0>(inf, l1_hi),
                            ldu([(4 << 24) | 1, 0, 0, 0]),
                        );
                        let (ab, abf) = fmin(a, af, b, bf);
                        fmin(ab, abf, e, ef)
                    };
                    (nc_lo, nf_lo, nc_hi, nf_hi) = (l, lf, h, hf);
                }
                c_lo = nc_lo;
                c_hi = nc_hi;
                vst1q_u32(fp, nf_lo);
                vst1q_u32(fp.add(4), nf_hi);
                src[i].write(src_bytes(nf_lo, nf_hi));
            }
            // The coded block flag: an all-zero block saves it.
            if let Some(cb) = cbf_base {
                let cc = cb + 1;
                let one = vdupq_n_f32(lam * self.bits(cc, 1));
                c_lo = vaddq_f32(c_lo, vsetq_lane_f32::<0>(lam * self.bits(cc, 0), one));
                c_hi = vaddq_f32(c_hi, one);
            }
            let mut cost = [0f32; L];
            vst1q_f32(cost.as_mut_ptr(), c_lo);
            vst1q_f32(cost.as_mut_ptr().add(4), c_hi);
            let mut lane = 0;
            for j in 1..L {
                if cost[j] < cost[lane] {
                    lane = j;
                }
            }
            let mut nz = 0;
            let mut at = (lane as u32) << 3;
            for i in 0..=last {
                // SAFETY: every position up to `last` was written above.
                let e = from[i].assume_init_ref()[(at >> 3) as usize & 7];
                nz += put_level(levels, coeffs, scan[start + i] as usize, e & 0xff_ffff);
                at = (src[i].assume_init() >> at) as u32 & 0x38;
            }
            nz
        }
    }
}

impl Trellis {
    /// [`Trellis::quant_n`] for a block whose rounded levels are all zero
    /// or one: the choices are then one or zero, so no path ever reaches a
    /// larger level's nodes (5..=8) and only the four lanes of nodes 0 and
    /// 2..=4 are live — one vector of four. The arithmetic is the general
    /// path's on those lanes, operation for operation, and the dead lanes
    /// there stay infinite and never win, so the levels are the same.
    #[allow(clippy::too_many_arguments)]
    fn quant_ones<const N: usize>(
        &self,
        kind: Block,
        coeffs: &[i32],
        levels: &mut [i16],
        x: &[f32; N],
        top: &[u32; N],
        last: usize,
        lam: f32,
    ) -> u32 {
        type V4 = [f32; 4];
        #[inline(always)]
        fn add4(a: V4, b: V4) -> V4 {
            [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
        }
        /// `from` code: the source lane (times eight, the bit offset of its
        /// code in the position's word) and the level (0 or 1).
        const fn c4(lane: u8, lv: u8) -> u8 {
            (lane << 3) | lv
        }
        const STAY4: [u8; 4] = [c4(0, 0), c4(1, 0), c4(2, 0), c4(3, 0)];
        let (scan, start, count): (&[u8], usize, usize) = match kind {
            Block::Luma8 => (&ZIGZAG8X8[..], 0, 64),
            Block::Luma4 => (&ZIGZAG4X4[..], 0, 16),
            _ => (&ZIGZAG4X4[..], 1, 15),
        };
        let (sig_base, last_base, abs_base, cbf_base) = kind.bases();
        // A level of one from each live lane: the first bin's context
        // increment is 1..=4 there (`FIRST_INC`), plus the sign.
        let one: V4 = std::array::from_fn(|j| lam * self.bits(abs_base + j + 1, 0) + lam);
        let max_idx = count - 1;
        let mut cost: V4 = [INF; 4];
        cost[0] = (last + 1..count).map(|i| x[i] * x[i]).sum();
        let mut from = [const { std::mem::MaybeUninit::<[u8; 4]>::uninit() }; N];
        for i in (0..=last).rev() {
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
            if top[i] == 0 {
                let zs0 = if coded_sig {
                    lam * self.bits(sc(), 0)
                } else {
                    0.0
                };
                cost = add4(add4(cost, [d0; 4]), [0.0, zs0, zs0, zs0]);
                from[i].write(STAY4);
                continue;
            }
            let (zs, os, l0, l1) = if coded_sig {
                let lc = if count == 64 {
                    last_base + LAST_COEFF_8X8_CTX[i] as usize
                } else {
                    last_base + i
                };
                let sc = sc();
                (
                    lam * self.bits(sc, 0),
                    lam * self.bits(sc, 1),
                    lam * self.bits(lc, 0),
                    lam * self.bits(lc, 1),
                )
            } else {
                (0.0, 0.0, 0.0, 0.0)
            };
            let sig: V4 = [os + l1, os + l0, os + l0, os + l0];
            let dd = (xi - 1.0) * (xi - 1.0);
            let l1v = add4(add4(add4(cost, [dd; 4]), sig), one);
            let z = add4(add4(cost, [d0; 4]), [0.0, zs, zs, zs]);
            // Lane 0 keeps its zero; a one moves lanes 0..=2 up one; then
            // lanes 1..=3's zeros, then lane 3's one (it saturates), each
            // only where strictly cheaper — the general path's order.
            let mut nc = [z[0], l1v[0], l1v[1], l1v[2]];
            let mut nf = [c4(0, 0), c4(0, 1), c4(1, 1), c4(2, 1)];
            for j in 1..4 {
                if z[j] < nc[j] {
                    nc[j] = z[j];
                    nf[j] = c4(j as u8, 0);
                }
            }
            if l1v[3] < nc[3] {
                nc[3] = l1v[3];
                nf[3] = c4(3, 1);
            }
            cost = nc;
            from[i].write(nf);
        }
        if let Some(cb) = cbf_base {
            let c = cb + 1;
            cost[0] += lam * self.bits(c, 0);
            let one = lam * self.bits(c, 1);
            for v in cost.iter_mut().skip(1) {
                *v += one;
            }
        }
        let mut lane = 0;
        for j in 1..4 {
            if cost[j] < cost[lane] {
                lane = j;
            }
        }
        // The walk back through the chosen path: a position's four codes
        // are one word, and the next lane's offset in it is the code at
        // this one's — a shift and a mask from one position to the next,
        // where an indexed load was four cycles more each.
        let mut nz = 0;
        let mut at = (lane as u32) << 3;
        for i in 0..=last {
            // SAFETY: every position up to `last` was written above.
            let e = u32::from_le_bytes(unsafe { from[i].assume_init() }) >> at;
            nz += put_level(levels, coeffs, scan[start + i] as usize, e & 1);
            at = e & 0x18;
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
            simd: Trellis::simd_default(),
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
                simd: Trellis::simd_default(),
            };
            let mut sum = 0u64;
            let start = std::time::Instant::now();
            // Evenly through the dump, not its first pictures only.
            let step = recs.len().div_ceil(20000).max(1);
            let recs: Vec<_> = recs.into_iter().step_by(step).collect();
            let cls = std::env::var("TRCLASS")
                .ok()
                .map(|v| v.parse::<u32>().unwrap());
            let recs: Vec<_> = recs
                .into_iter()
                .filter(|(kind, qbits, _, _, c, m)| {
                    let Some(cl) = cls else { return true };
                    let scale = 1.0 / (1u64 << qbits) as f32;
                    let st = if matches!(kind, Block::Luma16Ac | Block::ChromaAc) {
                        1
                    } else {
                        0
                    };
                    let mx = (st..c.len())
                        .map(|i| ((c[i].unsigned_abs() as f32) * m[i] as f32 * scale + 0.5) as u32)
                        .max()
                        .unwrap_or(0);
                    let k = if mx == 0 {
                        0
                    } else if mx == 1 {
                        1
                    } else {
                        2 + (*kind == Block::Luma8) as u32
                    };
                    k == cl
                })
                .collect();
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

    /// The NEON paths choose the scalar paths' levels: random blocks of
    /// every kind, at magnitudes that reach the zero-only, ones-only and
    /// general paths and the escape, under random context states and
    /// multipliers.
    #[test]
    fn simd_matches_scalar() {
        use crate::dsp::h264_enc::{Quant, qbits4, qbits8};
        use crate::h264::sps::ScalingLists;
        let lists = ScalingLists {
            list4x4: [[16; 16]; 6],
            list8x8: [[16; 64]; 6],
        };
        let q = Quant::new(&lists);
        let mut seed = 7u64;
        let mut rnd = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as u32
        };
        let kinds = [Block::Luma4, Block::Luma16Ac, Block::ChromaAc, Block::Luma8];
        let mut paths = [0u32; 3];
        for it in 0..20000 {
            let mut ctx = [0u8; 1024];
            for c in ctx.iter_mut() {
                *c = (rnd() % 126) as u8;
            }
            let qp = 12 + (rnd() % 40) as i32;
            let mut t = Trellis {
                ctx,
                lambda: crate::encode::h264_analyse::lambda_ssd(qp)
                    * (0.5 + (rnd() % 100) as f32 / 100.0),
                bd_off: 0,
                simd: false,
            };
            let kind = kinds[it % 4];
            let n = if kind == Block::Luma8 { 64 } else { 16 };
            let (mf, qbits) = if n == 64 {
                (&q.mf8[1][(qp % 6) as usize][..], qbits8(qp))
            } else {
                (&q.mf4[3][(qp % 6) as usize][..], qbits4(qp))
            };
            // A level scale of up to 0.6 (zeros), 1.4 (ones) or 40 steps.
            let step = (1u64 << qbits) as f32 / mf[0] as f32;
            let amp = [0.6f32, 1.4, 6.0, 40.0][(rnd() % 4) as usize] * step;
            let density = 1 + rnd() % 4;
            let mut c = vec![0i32; n];
            for v in c.iter_mut() {
                if rnd() % density == 0 {
                    let m = (rnd() as f32 / u32::MAX as f32) * amp;
                    *v = if rnd() & 1 == 0 {
                        m as i32
                    } else {
                        -(m as i32)
                    };
                }
            }
            let mut a = vec![0i16; n];
            let mut b = vec![0i16; n];
            let na = t.quant(kind, &c, &mut a, mf, qbits, qp);
            t.simd = true;
            let nb = t.quant(kind, &c, &mut b, mf, qbits, qp);
            assert_eq!((na, &a), (nb, &b), "kind {kind:?} qp {qp} coeffs {c:?}");
            let top = a.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0);
            paths[(top.min(2)) as usize] += 1;
        }
        // Every path was reached.
        assert!(paths.iter().all(|&p| p > 100), "{paths:?}");
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
