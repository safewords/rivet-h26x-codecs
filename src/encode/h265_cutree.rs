//! Lookahead quantiser propagation: how much of each block of a picture
//! the pictures coded after it inherit, and the quantiser offset that buys.
//!
//! # The idea
//!
//! A block that later pictures predict from is worth more than its own
//! bits: every picture that copies it inherits its quantisation error, and
//! every bit spent on it saves residual in each of them. So the encoder
//! looks ahead over the pictures it is holding, estimates for every block
//! of every one of them how much of its content comes from a reference —
//! one minus the ratio of its inter cost to its intra cost — and walks
//! that inheritance backwards through the coding order, block to the
//! blocks its motion points at, accumulating what each block passes on.
//! A block whose content is inherited heavily is quantised finer, by
//! `strength * log2(1 + inherited / own)` quantiser steps, where `own` is
//! its intra cost: the propagation model of the macroblock-tree idea
//! published for H.264 encoders, built here from that description on this
//! crate's own kernels.
//!
//! # What it measures with
//!
//! Every picture held is reduced once to half resolution (each sample the
//! mean of a 2x2 square, at 8 bits whatever the stream's depth) and
//! costed in 8x8 blocks of that — 16x16 of the picture, one quantisation
//! group. A block's intra cost is the least SATD of a flat, a horizontal
//! and a vertical prediction from its own picture's neighbouring samples;
//! its inter cost the least SATD of a small motion search against each
//! reference the coding structure will give it (and of the average of
//! both for a B picture), plus a small vector price. The searches are
//! cached per pair of pictures; each costs a few milliseconds at 720p.

use crate::dsp::distortion::DistortionDsp;
use crate::sample::Sample;

/// Half-resolution blocks are this many samples on a side.
const B: usize = 8;
/// The half-resolution planes are padded by this much on every side,
/// edge-replicated, and the search keeps inside it.
const PAD: usize = 24;
/// The search reaches this far around its best seed, half-resolution
/// samples.
const RANGE: i32 = 16;

/// One picture at half resolution, with its blocks' intra costs.
pub(crate) struct LowRes {
    /// Padded plane, `stride` wide.
    luma: Vec<u8>,
    stride: usize,
    /// Half-resolution size.
    w: usize,
    h: usize,
    /// Blocks across and down.
    pub bw: usize,
    pub bh: usize,
    /// Intra cost per block, raster.
    pub intra: Vec<u32>,
}

impl LowRes {
    /// Reduce the `w` by `h` luma plane `y` (samples at `bit_depth`).
    pub fn new<S: Sample>(
        y: &[S],
        w: usize,
        h: usize,
        bit_depth: u32,
        dist: &DistortionDsp<u8>,
    ) -> Self {
        let (lw, lh) = (w.div_ceil(2), h.div_ceil(2));
        let (bw, bh) = (lw.div_ceil(B), lh.div_ceil(B));
        let (pw, ph) = (bw * B, bh * B);
        let stride = pw + 2 * PAD;
        let mut luma = vec![0u8; stride * (ph + 2 * PAD)];
        let shift = bit_depth - 8;
        let at = |x: usize, y0: usize| y[y0.min(h - 1) * w + x.min(w - 1)].to_i32();
        for yy in 0..ph {
            for xx in 0..pw {
                let (x2, y2) = (2 * xx.min(lw - 1), 2 * yy.min(lh - 1));
                let s = at(x2, y2) + at(x2 + 1, y2) + at(x2, y2 + 1) + at(x2 + 1, y2 + 1);
                luma[(yy + PAD) * stride + xx + PAD] = (((s + 2) >> 2) >> shift).min(255) as u8;
            }
        }
        // Replicate the edges into the padding.
        for yy in PAD..PAD + ph {
            let row = yy * stride;
            let (l, r) = (luma[row + PAD], luma[row + PAD + pw - 1]);
            luma[row..row + PAD].fill(l);
            luma[row + PAD + pw..row + stride].fill(r);
        }
        for yy in 0..PAD {
            let (top, bot) = (PAD * stride, (PAD + ph - 1) * stride);
            luma.copy_within(top..top + stride, yy * stride);
            luma.copy_within(bot..bot + stride, (PAD + ph + yy) * stride);
        }
        let mut lr = LowRes {
            luma,
            stride,
            w: pw,
            h: ph,
            bw,
            bh,
            intra: vec![0; bw * bh],
        };
        lr.intra = (0..bw * bh).map(|b| lr.intra_cost(b, dist)).collect();
        lr
    }

    #[inline]
    fn off(&self, x: i32, y: i32) -> usize {
        ((y + PAD as i32) as usize) * self.stride + (x + PAD as i32) as usize
    }

    /// The least SATD of a flat, a horizontal and a vertical prediction of
    /// block `b` from the samples left of and above it.
    fn intra_cost(&self, b: usize, dist: &DistortionDsp<u8>) -> u32 {
        let (x, y) = (((b % self.bw) * B) as i32, ((b / self.bw) * B) as i32);
        let o = self.off(x, y);
        let src = &self.luma[o..];
        let top: Vec<i32> = (0..B)
            .map(|i| i32::from(self.luma[o - self.stride + i]))
            .collect();
        let left: Vec<i32> = (0..B)
            .map(|i| i32::from(self.luma[o + i * self.stride - 1]))
            .collect();
        let dc = ((top.iter().sum::<i32>() + left.iter().sum::<i32>() + B as i32) / (2 * B) as i32)
            as u8;
        let mut pred = [0u8; B * B];
        pred.fill(dc);
        let mut best = (dist.satd)(src, self.stride, &pred, B, B, B);
        for (i, p) in pred.iter_mut().enumerate() {
            *p = left[i / B] as u8;
        }
        best = best.min((dist.satd)(src, self.stride, &pred, B, B, B));
        for (i, p) in pred.iter_mut().enumerate() {
            *p = top[i % B] as u8;
        }
        best = best.min((dist.satd)(src, self.stride, &pred, B, B, B));
        // A block cannot be cheaper than a fixed floor: a perfectly flat
        // one still costs its signalling, and the ratios below divide by
        // this.
        best.max(B as u32 * 2)
    }
}

/// One picture's motion against one reference: per block, the cost and
/// the vector (half-resolution samples) of the best match.
pub(crate) struct Motion {
    pub cost: Vec<u32>,
    pub mv: Vec<(i16, i16)>,
}

/// Search every block of `cur` against `refp`.
pub(crate) fn search(cur: &LowRes, refp: &LowRes, dist: &DistortionDsp<u8>) -> Motion {
    let n = cur.bw * cur.bh;
    let mut cost = vec![0u32; n];
    let mut mv = vec![(0i16, 0i16); n];
    let lim = (PAD as i32) - 1;
    for b in 0..n {
        let (bx, by) = ((b % cur.bw) as i32, (b / cur.bw) as i32);
        let (x, y) = (bx * B as i32, by * B as i32);
        let src = &cur.luma[cur.off(x, y)..];
        let ok = |dx: i32, dy: i32| {
            x + dx >= -lim
                && y + dy >= -lim
                && x + dx + B as i32 <= refp.w as i32 + lim
                && y + dy + B as i32 <= refp.h as i32 + lim
        };
        let sad = |dx: i32, dy: i32| -> u32 {
            let o = refp.off(x + dx, y + dy);
            (dist.sad)(src, cur.stride, &refp.luma[o..], refp.stride, B, B)
                + 2 * (dx.unsigned_abs() + dy.unsigned_abs()).min(32)
        };
        // Seeds: zero and the vectors of the blocks left and above.
        let mut seeds = vec![(0i32, 0i32)];
        if bx > 0 {
            let m = mv[b - 1];
            seeds.push((i32::from(m.0), i32::from(m.1)));
        }
        if by > 0 {
            let m = mv[b - cur.bw];
            seeds.push((i32::from(m.0), i32::from(m.1)));
        }
        let mut best = (0i32, 0i32);
        let mut best_c = u32::MAX;
        for &(dx, dy) in &seeds {
            if ok(dx, dy) {
                let c = sad(dx, dy);
                if c < best_c {
                    best_c = c;
                    best = (dx, dy);
                }
            }
        }
        let centre = best;
        for step in [4i32, 2, 1] {
            loop {
                let from = best;
                for (sx, sy) in [(step, 0), (-step, 0), (0, step), (0, -step)] {
                    let (dx, dy) = (from.0 + sx, from.1 + sy);
                    if (dx - centre.0).abs() > RANGE || (dy - centre.1).abs() > RANGE || !ok(dx, dy)
                    {
                        continue;
                    }
                    let c = sad(dx, dy);
                    if c < best_c {
                        best_c = c;
                        best = (dx, dy);
                    }
                }
                if best == from {
                    break;
                }
            }
        }
        let o = refp.off(x + best.0, y + best.1);
        cost[b] = (dist.satd)(src, cur.stride, &refp.luma[o..], refp.stride, B, B);
        mv[b] = (best.0 as i16, best.1 as i16);
    }
    Motion { cost, mv }
}

/// The SATD of block `b` of `cur` against the average of its two motion
/// compensated predictions.
pub(crate) fn bi_cost(
    cur: &LowRes,
    r0: &LowRes,
    m0: (i16, i16),
    r1: &LowRes,
    m1: (i16, i16),
    b: usize,
    dist: &DistortionDsp<u8>,
) -> u32 {
    let (x, y) = (((b % cur.bw) * B) as i32, ((b / cur.bw) * B) as i32);
    let o0 = r0.off(x + i32::from(m0.0), y + i32::from(m0.1));
    let o1 = r1.off(x + i32::from(m1.0), y + i32::from(m1.1));
    let mut pred = [0u8; B * B];
    for yy in 0..B {
        for xx in 0..B {
            pred[yy * B + xx] = ((u16::from(r0.luma[o0 + yy * r0.stride + xx])
                + u16::from(r1.luma[o1 + yy * r1.stride + xx])
                + 1)
                >> 1) as u8;
        }
    }
    (dist.satd)(&cur.luma[cur.off(x, y)..], cur.stride, &pred, B, B, B)
}

/// Pass `amount` from block `b` of a picture to the blocks of its
/// reference that the block's displaced copy overlaps, in proportion to
/// the overlap.
pub(crate) fn spread(
    into: &mut [f32],
    bw: usize,
    bh: usize,
    b: usize,
    mv: (i16, i16),
    amount: f32,
) {
    let x = ((b % bw) * B) as i32 + i32::from(mv.0);
    let y = ((b / bw) * B) as i32 + i32::from(mv.1);
    let (bx, by) = (x.div_euclid(B as i32), y.div_euclid(B as i32));
    let (fx, fy) = (x.rem_euclid(B as i32), y.rem_euclid(B as i32));
    let area = (B * B) as f32;
    for (dx, wx) in [(0, B as i32 - fx), (1, fx)] {
        for (dy, wy) in [(0, B as i32 - fy), (1, fy)] {
            let (tx, ty) = (bx + dx, by + dy);
            if wx == 0 || wy == 0 || tx < 0 || ty < 0 || tx >= bw as i32 || ty >= bh as i32 {
                continue;
            }
            into[ty as usize * bw + tx as usize] += amount * (wx * wy) as f32 / area;
        }
    }
}

/// The quantiser offsets of a picture whose blocks have intra costs
/// `intra` and inherited amounts `inherited`, at `strength`, laid out per
/// quantisation group of `1 << log2_qg` over a coded picture of `cw` by
/// `ch` (16x16 groups take one block each; larger ones average theirs).
pub(crate) fn offsets(
    lr: &LowRes,
    inherited: &[f32],
    strength: f32,
    cw: usize,
    ch: usize,
    log2_qg: u32,
) -> Vec<f32> {
    let per = |b: usize| -> f32 {
        let own = lr.intra[b] as f32;
        -strength * (1.0 + inherited[b] / own).log2()
    };
    let qg = 1usize << log2_qg;
    let (wq, hq) = (cw.div_ceil(qg), ch.div_ceil(qg));
    let mut out = vec![0f32; wq * hq];
    for gy in 0..hq {
        for gx in 0..wq {
            // Blocks of 16 picture samples covered by this group.
            let (x0, y0) = (gx * qg / 16, gy * qg / 16);
            let k = (qg / 16).max(1);
            let mut sum = 0f32;
            let mut cnt = 0f32;
            for by in y0..(y0 + k).min(lr.bh) {
                for bx in x0..(x0 + k).min(lr.bw) {
                    sum += per(by * lr.bw + bx);
                    cnt += 1.0;
                }
            }
            if cnt > 0.0 {
                out[gy * wq + gx] = sum / cnt;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A block copied unchanged into the next picture passes its whole
    /// intra cost on through `spread`, and the offsets fall with it; a
    /// block nothing inherits keeps offset zero.
    #[test]
    fn inheritance_lowers_the_quantiser() {
        let dist = DistortionDsp::<u8>::scalar();
        let (w, h) = (64usize, 64usize);
        let mut s = 1u32;
        let y: Vec<u8> = (0..w * h)
            .map(|_| {
                s = s.wrapping_mul(1103515245).wrapping_add(12345);
                (s >> 24) as u8
            })
            .collect();
        let a = LowRes::new(&y, w, h, 8, &dist);
        let m = search(&a, &a, &dist);
        assert!(
            m.cost.iter().all(|&c| c == 0),
            "a picture matches itself at zero cost"
        );
        assert!(m.mv.iter().all(|&v| v == (0, 0)));
        let mut into = vec![0f32; a.bw * a.bh];
        for b in 0..a.bw * a.bh {
            let own = a.intra[b] as f32;
            spread(&mut into, a.bw, a.bh, b, m.mv[b], own);
        }
        let off = offsets(&a, &into, 2.0, w, h, 4);
        assert!(off.iter().all(|&o| (o + 2.0).abs() < 1e-4), "{off:?}");
        let none = offsets(&a, &vec![0f32; a.bw * a.bh], 2.0, w, h, 4);
        assert!(none.iter().all(|&o| o == 0.0));
    }
}
