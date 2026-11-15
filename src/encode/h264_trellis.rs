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

/// Path states: 0 nothing coded yet; 1..=4 ones seen 0..=3+ with no larger
/// level; 5..=8 larger levels seen 1..=4+.
const NONE: usize = 0;
const NODES: usize = 9;

#[inline]
fn first_inc(node: usize) -> usize {
    match node {
        0 => 1,
        1..=4 => (node).min(4), // 1 + eq1, eq1 = node - 1
        _ => 0,
    }
}

#[inline]
fn rest_inc(node: usize) -> usize {
    if node >= 5 { 5 + (node - 4).min(4) } else { 5 }
}

#[inline]
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
        let mut x = [0f32; N];
        let mut top = [0u32; N];
        let mut last = usize::MAX;
        for i in 0..count {
            let r = scan[start + i] as usize;
            let v = (coeffs[r].unsigned_abs() as f32) * mf[r] as f32 * scale;
            x[i] = v;
            let t = (v + 0.5) as u32;
            top[i] = t;
            if t != 0 {
                last = i;
            }
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
        // Per node: the cost of a level of 1, and the fixed part of a larger
        // one (first bin 1 plus the rest-context terminator) and the cost per
        // extra unit, all with the sign.
        let mut one = [0f32; NODES];
        let mut big0 = [0f32; NODES];
        let mut big1 = [0f32; NODES];
        for n in 0..NODES {
            let f = first_inc(n);
            let r = rest_inc(n);
            one[n] = lb[f][0] + lam;
            big0[n] = lb[f][1] + lb[r][0] + lam;
            big1[n] = lb[r][1];
        }
        let level_cost = |n: usize, abs: u32| -> f32 {
            if abs == 1 {
                one[n]
            } else if abs < 15 {
                big0[n] + (abs - 2) as f32 * big1[n]
            } else {
                // A capped prefix has no terminator; the escape is bypass.
                let mut rem = abs - 15;
                let mut k = 0;
                while rem >= (1 << k) {
                    rem -= 1 << k;
                    k += 1;
                }
                big0[n] - lb[rest_inc(n)][0] + 13.0 * big1[n] + lam * (2 * k + 1) as f32
            }
        };
        let max_idx = count - 1;

        let mut cost = [f32::INFINITY; NODES];
        // Everything past `last` is zero and costs its squared error only.
        cost[NONE] = (last + 1..count).map(|i| x[i] * x[i]).sum();
        let mut live: u16 = 1 << NONE;
        // Per position and resulting node: the node it came from and the
        // level chosen, packed.
        let mut from = [[0u32; NODES]; N];
        for i in (0..=last).rev() {
            let mut nc = [f32::INFINITY; NODES];
            let mut nf = [0u32; NODES];
            let mut nlive: u16 = 0;
            let t = top[i];
            let coded_sig = i < max_idx;
            let xi = x[i];
            let d0 = xi * xi;
            if t == 0 {
                let zs0 = if coded_sig {
                    let sc = if count == 64 {
                        sig_base + SIG_COEFF_8X8_CTX[0][i] as usize
                    } else {
                        sig_base + i
                    };
                    lam * self.bits(sc, 0)
                } else {
                    0.0
                };
                // Only zero is on offer: every path stays on its node.
                let mut m = live;
                let mut nf = [0u32; NODES];
                while m != 0 {
                    let node = m.trailing_zeros() as usize;
                    m &= m - 1;
                    cost[node] = cost[node] + d0 + if node == NONE { 0.0 } else { zs0 };
                    nf[node] = (node as u32) << 24;
                }
                from[i] = nf;
                continue;
            }
            let (zs, os, l0, l1) = if coded_sig {
                let sc = if count == 64 {
                    sig_base + SIG_COEFF_8X8_CTX[0][i] as usize
                } else {
                    sig_base + i
                };
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
            let mut m = live;
            while m != 0 {
                let node = m.trailing_zeros() as usize;
                m &= m - 1;
                let c = cost[node];
                // Zero.
                if t <= 2 {
                    let j = c + d0 + if node == NONE { 0.0 } else { zs };
                    if j < nc[node] {
                        nc[node] = j;
                        nf[node] = (node as u32) << 24;
                        nlive |= 1 << node;
                    }
                }
                // The rounded level and one less.
                let sig = os + if node == NONE { l1 } else { l0 };
                let lo = t.saturating_sub(1).max(1);
                for lv in lo..=t {
                    if lv == 0 {
                        continue;
                    }
                    let d = xi - lv as f32;
                    let j = c + d * d + sig + level_cost(node, lv);
                    let to = next(node, lv);
                    if j < nc[to] {
                        nc[to] = j;
                        nf[to] = ((node as u32) << 24) | lv;
                        nlive |= 1 << to;
                    }
                }
            }
            cost = nc;
            live = nlive;
            from[i] = nf;
        }
        // The coded block flag: an all-zero block saves it.
        if let Some(cb) = cbf_base {
            let c = cb + 1;
            cost[NONE] += lam * self.bits(c, 0);
            let one = lam * self.bits(c, 1);
            for v in cost.iter_mut().skip(1) {
                *v += one;
            }
        }
        let mut node = 0;
        for n in 1..NODES {
            if cost[n] < cost[node] {
                node = n;
            }
        }
        let mut nz = 0;
        for i in 0..=last {
            let e = from[i][node];
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
            node = (e >> 24) as usize;
        }
        nz
    }
}
