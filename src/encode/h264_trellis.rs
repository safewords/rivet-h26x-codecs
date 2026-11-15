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

static TLAM: std::sync::LazyLock<f32> = std::sync::LazyLock::new(|| {
    std::env::var("H26X_TLAM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0)
});

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

    /// Bits of `coeff_abs_level_minus1` and the sign for a level of
    /// magnitude `abs` coded in path state `node`.
    fn level_bits(&self, abs_base: usize, node: usize, abs: u32) -> f32 {
        let c0 = abs_base + first_inc(node);
        if abs == 1 {
            return self.bits(c0, 0) + 1.0;
        }
        let c1 = abs_base + rest_inc(node);
        let m1 = abs - 1;
        let prefix = m1.min(14);
        let mut b = self.bits(c0, 1) + (prefix - 1) as f32 * self.bits(c1, 1);
        if prefix < 14 {
            b += self.bits(c1, 0);
        } else {
            let mut rem = m1 - 14;
            let mut k = 0;
            while rem >= (1 << k) {
                rem -= 1 << k;
                k += 1;
            }
            b += (2 * k + 1) as f32;
        }
        b + 1.0
    }

    /// Quantise `coeffs` (raster, `n` = 16 or 64) into `levels` by the
    /// trellis: `mf` and `qbits` the quantiser's, `qp` the block's own
    /// quantiser, primed (for its step). AC blocks leave position 0 at zero (its
    /// DC is coded elsewhere). Returns the nonzero count.
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
        let (scan, start, count): (&[u8], usize, usize) = match kind {
            Block::Luma8 => (&ZIGZAG8X8[..], 0, 64),
            Block::Luma4 => (&ZIGZAG4X4[..], 0, 16),
            _ => (&ZIGZAG4X4[..], 1, 15),
        };
        let (sig_base, last_base, abs_base, cbf_base) = kind.bases();
        // lambda in this block's level units: squared error scales with its
        // own step (chroma's differs from luma's).
        let own = qstep(qp - self.bd_off);
        let lam = self.lambda / (own * own) * *TLAM;
        let scale = 1.0 / (1u64 << qbits) as f32;

        // Exact levels and their rounded magnitudes, in scan order.
        let mut x = [0f32; 64];
        let mut top = [0u32; 64];
        let mut last = None;
        for i in 0..count {
            let r = scan[start + i] as usize;
            x[i] = (coeffs[r].unsigned_abs() as f32) * mf[r] as f32 * scale;
            top[i] = (x[i] + 0.5) as u32;
            if top[i] != 0 {
                last = Some(i);
            }
        }
        for l in levels.iter_mut().take(if count == 64 { 64 } else { 16 }) {
            *l = 0;
        }
        let Some(last) = last else {
            return 0;
        };
        let max_idx = count - 1;
        let sig_ctx = |i: usize| {
            if count == 64 {
                sig_base + SIG_COEFF_8X8_CTX[0][i] as usize
            } else {
                sig_base + i
            }
        };
        let last_ctx = |i: usize| {
            if count == 64 {
                last_base + LAST_COEFF_8X8_CTX[i] as usize
            } else {
                last_base + i
            }
        };

        let mut cost = [f32::INFINITY; NODES];
        cost[NONE] = 0.0;
        // Per position and node: (from node, level) for the backtrack.
        let mut from = [[(0u8, 0u32); NODES]; 64];
        // Everything past `last` is zero and costs its squared error only.
        let tail: f32 = (last + 1..count).map(|i| x[i] * x[i]).sum();
        cost[NONE] += tail;
        for i in (0..=last).rev() {
            let mut nc = [f32::INFINITY; NODES];
            let mut nf = [(0u8, 0u32); NODES];
            let t = top[i];
            let cands: [u32; 3] = [t, t.saturating_sub(1), 0];
            let ncand = if t == 0 {
                1
            } else if t == 1 {
                2
            } else if t <= 2 {
                3
            } else {
                2
            };
            let coded_sig = i < max_idx;
            let (zs, os) = if coded_sig {
                (self.bits(sig_ctx(i), 0), self.bits(sig_ctx(i), 1))
            } else {
                (0.0, 0.0)
            };
            let (l0, l1) = if coded_sig {
                (self.bits(last_ctx(i), 0), self.bits(last_ctx(i), 1))
            } else {
                (0.0, 0.0)
            };
            for (node, &c) in cost.iter().enumerate() {
                if !c.is_finite() {
                    continue;
                }
                for &lv in cands.iter().skip(if t == 0 { 2 } else { 0 }).take(ncand) {
                    let d = x[i] - lv as f32;
                    let (to, bits) = if lv == 0 {
                        let b = if node == NONE { 0.0 } else { zs };
                        (node, b)
                    } else {
                        let b = os
                            + if node == NONE { l1 } else { l0 }
                            + self.level_bits(abs_base, node, lv);
                        (next(node, lv), b)
                    };
                    let j = c + d * d + lam * bits;
                    if j < nc[to] {
                        nc[to] = j;
                        nf[to] = (node as u8, lv);
                    }
                }
            }
            cost = nc;
            from[i] = nf;
        }
        // The coded block flag: an all-zero block saves it.
        if let Some(cb) = cbf_base {
            let c = cb + 1;
            cost[NONE] += lam * self.bits(c, 0);
            for v in cost.iter_mut().skip(1) {
                *v += lam * self.bits(c, 1);
            }
        }
        let mut node = (0..NODES)
            .min_by(|&a, &b| cost[a].partial_cmp(&cost[b]).unwrap())
            .unwrap();
        let mut nz = 0;
        for i in 0..=last {
            let (prev, lv) = from[i][node];
            if lv != 0 {
                let r = scan[start + i] as usize;
                levels[r] = if coeffs[r] < 0 {
                    -(lv as i16)
                } else {
                    lv as i16
                };
                nz += 1;
            }
            node = prev as usize;
        }
        nz
    }
}
