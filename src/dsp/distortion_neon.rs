//! AArch64 NEON versions of the distortion metrics, for 8-bit samples —
//! the same kernels as [`super::distortion_x86`], with the same lane
//! layout, on the instructions this architecture has for them.
//!
//! - SAD: `uabd` gives the absolute byte differences, `uadalp` folds
//!   them pairwise into wider lanes. A row's differences are summed in u16
//!   (at most four sixteen-byte chunks of 510 per lane) and folded into a
//!   u32 accumulator once a row, so no width is ever close to overflow.
//! - SSD: `uabdl` widens the differences to u16, `umlal` squares and
//!   accumulates them in u32. A 64x64 block puts at most 66.6 million in
//!   a lane.
//! - SATD: two 4x4 tiles per vector, rows as i16 from `usubl` (a wrapped
//!   u16 difference reinterpreted, which is the signed difference exactly
//!   because it is at most 255 in magnitude). Butterflies lane-wise, a
//!   transpose of each tile by `trn1`/`trn2` at 16 and then 32 bits —
//!   which transposes both halves at once, since `trn` pairs lanes across
//!   the whole vector — butterflies again, `abs`, and the per-tile
//!   `(sum + 1) >> 1` after `saddlp` and `addp` have widened and folded
//!   the halves.
//!
//! Written on x86 and checked for compilation against
//! `aarch64-unknown-linux-gnu`; the bit-exactness test below is the same
//! one the x86 module runs and executes on the CI arm64 runners (a PR
//! against develop runs it on Linux and macOS arm64). `install` takes
//! the table for every `cpu.neon`, which on AArch64 is every CPU.

#![cfg(target_arch = "aarch64")]

use std::arch::aarch64::*;

use super::Cpu;
use super::distortion::{DistortionDsp, WpMoments, sad_scalar, satd_scalar, ssd_scalar, weighted_sad_scalar, wp_moments_scalar};

/// Four bytes at `p` in the low lanes of a vector, the rest zero.
#[inline(always)]
unsafe fn load4(p: *const u8) -> uint8x8_t {
    unsafe { vreinterpret_u8_u32(vcreate_u32((p as *const u32).read_unaligned() as u64)) }
}

/// Four bytes at `p` and four at `q`, in the low and high halves.
#[inline(always)]
unsafe fn load4x2(p: *const u8, q: *const u8) -> uint8x8_t {
    unsafe {
        let lo = (p as *const u32).read_unaligned() as u64;
        let hi = (q as *const u32).read_unaligned() as u64;
        vreinterpret_u8_u64(vcreate_u64(lo | (hi << 32)))
    }
}

#[target_feature(enable = "neon")]
unsafe fn sad_impl(a: *const u8, sa: usize, b: *const u8, sb: usize, w: usize, h: usize) -> u32 {
    unsafe {
        let mut acc = vdupq_n_u32(0);
        for y in 0..h {
            let ra = a.add(y * sa);
            let rb = b.add(y * sb);
            let mut row = vdupq_n_u16(0);
            let mut x = 0;
            while x + 16 <= w {
                let d = vabdq_u8(vld1q_u8(ra.add(x)), vld1q_u8(rb.add(x)));
                row = vpadalq_u8(row, d);
                x += 16;
            }
            if x + 8 <= w {
                row = vaddq_u16(row, vabdl_u8(vld1_u8(ra.add(x)), vld1_u8(rb.add(x))));
                x += 8;
            }
            if x + 4 <= w {
                row = vaddq_u16(row, vabdl_u8(load4(ra.add(x)), load4(rb.add(x))));
            }
            acc = vpadalq_u16(acc, row);
        }
        vaddvq_u32(acc)
    }
}

pub(crate) fn sad(a: &[u8], a_stride: usize, b: &[u8], b_stride: usize, w: usize, h: usize) -> u32 {
    if w % 4 != 0 || h == 0 {
        return sad_scalar(a, a_stride, b, b_stride, w, h);
    }
    assert!(a.len() >= (h - 1) * a_stride + w && b.len() >= (h - 1) * b_stride + w, "block out of range");
    unsafe { sad_impl(a.as_ptr(), a_stride, b.as_ptr(), b_stride, w, h) }
}

#[target_feature(enable = "neon")]
unsafe fn ssd_impl(a: *const u8, sa: usize, b: *const u8, sb: usize, w: usize, h: usize) -> u64 {
    unsafe {
        let mut acc = vdupq_n_u32(0);
        let mut sq = |d: uint16x8_t| {
            acc = vmlal_u16(acc, vget_low_u16(d), vget_low_u16(d));
            acc = vmlal_u16(acc, vget_high_u16(d), vget_high_u16(d));
        };
        for y in 0..h {
            let ra = a.add(y * sa);
            let rb = b.add(y * sb);
            let mut x = 0;
            while x + 16 <= w {
                let va = vld1q_u8(ra.add(x));
                let vb = vld1q_u8(rb.add(x));
                sq(vabdl_u8(vget_low_u8(va), vget_low_u8(vb)));
                sq(vabdl_u8(vget_high_u8(va), vget_high_u8(vb)));
                x += 16;
            }
            if x + 8 <= w {
                sq(vabdl_u8(vld1_u8(ra.add(x)), vld1_u8(rb.add(x))));
                x += 8;
            }
            if x + 4 <= w {
                sq(vabdl_u8(load4(ra.add(x)), load4(rb.add(x))));
            }
        }
        vaddlvq_u32(acc)
    }
}

pub(crate) fn ssd(a: &[u8], a_stride: usize, b: &[u8], b_stride: usize, w: usize, h: usize) -> u64 {
    if w % 4 != 0 || h == 0 {
        return ssd_scalar(a, a_stride, b, b_stride, w, h);
    }
    assert!(a.len() >= (h - 1) * a_stride + w && b.len() >= (h - 1) * b_stride + w, "block out of range");
    unsafe { ssd_impl(a.as_ptr(), a_stride, b.as_ptr(), b_stride, w, h) }
}

/// The 4-point Hadamard butterfly, lane-wise across four vectors.
#[inline(always)]
unsafe fn butterfly(r0: int16x8_t, r1: int16x8_t, r2: int16x8_t, r3: int16x8_t) -> [int16x8_t; 4] {
    unsafe {
        let s0 = vaddq_s16(r0, r3);
        let s1 = vaddq_s16(r1, r2);
        let s2 = vsubq_s16(r1, r2);
        let s3 = vsubq_s16(r0, r3);
        [vaddq_s16(s0, s1), vaddq_s16(s3, s2), vsubq_s16(s0, s1), vsubq_s16(s3, s2)]
    }
}

/// SATD of the two tiles in `r0..r3` (one row each, A low, B high), as
/// `[A, B, A, B]` i32 with the per-tile rounding applied.
#[inline(always)]
unsafe fn satd_pair(r0: int16x8_t, r1: int16x8_t, r2: int16x8_t, r3: int16x8_t) -> int32x4_t {
    unsafe {
        let [t0, t1, t2, t3] = butterfly(r0, r1, r2, r3);
        // Transpose each 4x4 tile: 16-bit pairs, then 32-bit pairs.
        let a0 = vtrn1q_s16(t0, t1);
        let a1 = vtrn2q_s16(t0, t1);
        let a2 = vtrn1q_s16(t2, t3);
        let a3 = vtrn2q_s16(t2, t3);
        let c0 = vreinterpretq_s16_s32(vtrn1q_s32(vreinterpretq_s32_s16(a0), vreinterpretq_s32_s16(a2)));
        let c2 = vreinterpretq_s16_s32(vtrn2q_s32(vreinterpretq_s32_s16(a0), vreinterpretq_s32_s16(a2)));
        let c1 = vreinterpretq_s16_s32(vtrn1q_s32(vreinterpretq_s32_s16(a1), vreinterpretq_s32_s16(a3)));
        let c3 = vreinterpretq_s16_s32(vtrn2q_s32(vreinterpretq_s32_s16(a1), vreinterpretq_s32_s16(a3)));
        let [w0, w1, w2, w3] = butterfly(c0, c1, c2, c3);
        let s = vaddq_s16(vaddq_s16(vabsq_s16(w0), vabsq_s16(w1)), vaddq_s16(vabsq_s16(w2), vabsq_s16(w3)));
        // [A01, A23, B01, B23] -> [A, B, A, B], then the tile rounding.
        let p = vpaddlq_s16(s);
        let q = vpaddq_s32(p, p);
        vshrq_n_s32(vaddq_s32(q, vdupq_n_s32(1)), 1)
    }
}

#[inline(always)]
unsafe fn diff8(a: uint8x8_t, b: uint8x8_t) -> int16x8_t {
    unsafe { vreinterpretq_s16_u16(vsubl_u8(a, b)) }
}

#[target_feature(enable = "neon")]
unsafe fn satd_impl(a: *const u8, sa: usize, b: *const u8, sb: usize, w: usize, h: usize) -> u32 {
    unsafe {
        let mut acc = vdupq_n_s32(0);
        if w == 4 {
            let mut y = 0;
            while y + 8 <= h {
                let row = |r: usize| {
                    diff8(load4x2(a.add((y + r) * sa), a.add((y + r + 4) * sa)), load4x2(b.add((y + r) * sb), b.add((y + r + 4) * sb)))
                };
                acc = vaddq_s32(acc, satd_pair(row(0), row(1), row(2), row(3)));
                y += 8;
            }
            if y < h {
                let row = |r: usize| diff8(load4(a.add((y + r) * sa)), load4(b.add((y + r) * sb)));
                acc = vaddq_s32(acc, satd_pair(row(0), row(1), row(2), row(3)));
            }
        } else {
            let mut y = 0;
            while y < h {
                let ra = a.add(y * sa);
                let rb = b.add(y * sb);
                let mut x = 0;
                while x + 8 <= w {
                    let row = |r: usize| diff8(vld1_u8(ra.add(r * sa + x)), vld1_u8(rb.add(r * sb + x)));
                    acc = vaddq_s32(acc, satd_pair(row(0), row(1), row(2), row(3)));
                    x += 8;
                }
                if x < w {
                    let row = |r: usize| diff8(load4(ra.add(r * sa + x)), load4(rb.add(r * sb + x)));
                    acc = vaddq_s32(acc, satd_pair(row(0), row(1), row(2), row(3)));
                }
                y += 4;
            }
        }
        // Lanes are [A, B, A, B] sums: one of each.
        (vgetq_lane_s32(acc, 0) as u32).wrapping_add(vgetq_lane_s32(acc, 1) as u32)
    }
}

pub(crate) fn satd(a: &[u8], a_stride: usize, b: &[u8], b_stride: usize, w: usize, h: usize) -> u32 {
    if w % 4 != 0 || h % 4 != 0 || h == 0 {
        return satd_scalar(a, a_stride, b, b_stride, w, h);
    }
    assert!(a.len() >= (h - 1) * a_stride + w && b.len() >= (h - 1) * b_stride + w, "block out of range");
    unsafe { satd_impl(a.as_ptr(), a_stride, b.as_ptr(), b_stride, w, h) }
}

// ----------------------------------------------------------------------
// Weighted-prediction fit: moments and weighted SAD (8-bit)
// ----------------------------------------------------------------------

/// 16 samples a step: `uadalp` folds the bytes (and their `uabd`) into u32
/// lanes for the plain sums and the SAD, `umull` squares / multiplies them
/// into u16 (at most 255^2) and `uadalp` folds those into u32 too; a row of
/// up to 2^14 samples puts under 2^29 in a lane, and the lanes are folded
/// into u64 once a row.
pub(crate) fn wp_moments(cur: &[u8], cur_stride: usize, refp: &[u8], ref_stride: usize, w: usize, h: usize) -> WpMoments {
    if w == 0 || h == 0 || w > 1 << 14 {
        return wp_moments_scalar(cur, cur_stride, refp, ref_stride, w, h);
    }
    assert!(cur.len() >= (h - 1) * cur_stride + w && refp.len() >= (h - 1) * ref_stride + w, "region out of range");
    // SAFETY: NEON is baseline on AArch64; every load is inside the rows
    // checked above.
    unsafe {
        let mut acc = [vdupq_n_u64(0); 5];
        let mut m = WpMoments::default();
        for y in 0..h {
            let (c, r) = (cur.as_ptr().add(y * cur_stride), refp.as_ptr().add(y * ref_stride));
            let mut row = [vdupq_n_u32(0); 5];
            let mut x = 0;
            while x + 16 <= w {
                let vr = vld1q_u8(r.add(x));
                let vc = vld1q_u8(c.add(x));
                row[0] = vpadalq_u16(row[0], vpaddlq_u8(vr));
                row[1] = vpadalq_u16(row[1], vpaddlq_u8(vc));
                row[2] = vpadalq_u16(row[2], vmull_u8(vget_low_u8(vr), vget_low_u8(vr)));
                row[2] = vpadalq_u16(row[2], vmull_high_u8(vr, vr));
                row[3] = vpadalq_u16(row[3], vmull_u8(vget_low_u8(vr), vget_low_u8(vc)));
                row[3] = vpadalq_u16(row[3], vmull_high_u8(vr, vc));
                row[4] = vpadalq_u16(row[4], vpaddlq_u8(vabdq_u8(vr, vc)));
                x += 16;
            }
            for (a, v) in acc.iter_mut().zip(row) {
                *a = vpadalq_u32(*a, v);
            }
            while x < w {
                let (r, c) = (u64::from(*r.add(x)), u64::from(*c.add(x)));
                m.sr += r;
                m.sc += c;
                m.srr += r * r;
                m.src += r * c;
                m.sad += r.abs_diff(c);
                x += 1;
            }
        }
        m.sr += vaddvq_u64(acc[0]);
        m.sc += vaddvq_u64(acc[1]);
        m.srr += vaddvq_u64(acc[2]);
        m.src += vaddvq_u64(acc[3]);
        m.sad += vaddvq_u64(acc[4]);
        m
    }
}

/// 16 samples a step, in i32 as the reference has it: `smull` by the
/// weight, the rounding term, `sshl` by minus the shift (an arithmetic
/// shift right), the offset, the clip to 0..=255, `sabd` against the
/// source, `uadalp` into the row's sums (at most 510 a lane a step).
#[allow(clippy::too_many_arguments)]
pub(crate) fn weighted_sad(cur: &[u8], cur_stride: usize, refp: &[u8], ref_stride: usize, w: usize, h: usize, weight: i32, shift: u32, offset: i32, max: i32) -> u64 {
    if w == 0 || h == 0 || w > 1 << 20 || !(-32768..=32767).contains(&weight) || shift > 14 || max != 255 {
        return weighted_sad_scalar(cur, cur_stride, refp, ref_stride, w, h, weight, shift, offset, max);
    }
    assert!(cur.len() >= (h - 1) * cur_stride + w && refp.len() >= (h - 1) * ref_stride + w, "region out of range");
    let round = if shift >= 1 { 1i32 << (shift - 1) } else { 0 };
    // SAFETY: NEON is baseline on AArch64; every load is inside the rows
    // checked above.
    unsafe {
        let (rnd, sh, off) = (vdupq_n_s32(round), vdupq_n_s32(-(shift as i32)), vdupq_n_s32(offset));
        let (lo, hi) = (vdupq_n_s32(0), vdupq_n_s32(255));
        let wt = weight as i16;
        let predict = |r: int16x4_t| vminq_s32(vmaxq_s32(vaddq_s32(vshlq_s32(vaddq_s32(vmull_n_s16(r, wt), rnd), sh), off), lo), hi);
        let mut acc = vdupq_n_u64(0);
        let mut total = 0u64;
        for y in 0..h {
            let (c, r) = (cur.as_ptr().add(y * cur_stride), refp.as_ptr().add(y * ref_stride));
            let mut row = vdupq_n_u32(0);
            let mut x = 0;
            while x + 16 <= w {
                let vr = vld1q_u8(r.add(x));
                let vc = vld1q_u8(c.add(x));
                for (r8, c8) in [(vget_low_u8(vr), vget_low_u8(vc)), (vget_high_u8(vr), vget_high_u8(vc))] {
                    let (r16, c16) = (vreinterpretq_s16_u16(vmovl_u8(r8)), vreinterpretq_s16_u16(vmovl_u8(c8)));
                    let da = vabdq_s32(predict(vget_low_s16(r16)), vmovl_s16(vget_low_s16(c16)));
                    let db = vabdq_s32(predict(vget_high_s16(r16)), vmovl_s16(vget_high_s16(c16)));
                    row = vaddq_u32(row, vaddq_u32(vreinterpretq_u32_s32(da), vreinterpretq_u32_s32(db)));
                }
                x += 16;
            }
            acc = vpadalq_u32(acc, row);
            while x < w {
                let p = ((i32::from(*r.add(x)) * weight + round) >> shift) + offset;
                total += u64::from(p.clamp(0, 255).abs_diff(i32::from(*c.add(x))));
                x += 1;
            }
        }
        total + vaddvq_u64(acc)
    }
}

/// Install the NEON kernels.
pub fn install(d: &mut DistortionDsp<u8>, cpu: Cpu) {
    if cpu.neon {
        d.sad = sad;
        d.satd = satd;
        d.ssd = ssd;
        d.wp_moments = wp_moments;
        d.weighted_sad = weighted_sad;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> u32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*seed >> 33) as u32
    }

    const SIZES: [(usize, usize); 16] = [
        (4, 4),
        (4, 8),
        (4, 16),
        (8, 4),
        (8, 8),
        (8, 16),
        (12, 8),
        (12, 12),
        (16, 4),
        (16, 8),
        (16, 16),
        (20, 8),
        (24, 16),
        (32, 32),
        (48, 16),
        (64, 64),
    ];

    fn planes(seed: &mut u64) -> (Vec<u8>, Vec<u8>) {
        let n = 96 * 96;
        let mut a = vec![0u8; n];
        let mut b = vec![0u8; n];
        for y in 0..96 {
            let mode = lcg(seed) % 8;
            for x in 0..96 {
                let (va, vb) = match mode {
                    0 => (0, 255),
                    1 => (255, 0),
                    2 => (lcg(seed) as u8, 255),
                    _ => (lcg(seed) as u8, lcg(seed) as u8),
                };
                a[y * 96 + x] = va;
                b[y * 96 + x] = vb;
            }
        }
        (a, b)
    }

    #[test]
    fn neon_matches_scalar() {
        let s = DistortionDsp::<u8>::scalar();
        let mut d = DistortionDsp::<u8>::scalar();
        install(&mut d, Cpu { neon: true, ..Cpu::SCALAR });
        let mut seed = 0x5add_u64;
        for round in 0..24 {
            let (a, b) = planes(&mut seed);
            for &(w, h) in &SIZES {
                let sa = w + (lcg(&mut seed) as usize % 24);
                let sb = w + (lcg(&mut seed) as usize % 24);
                let oa = lcg(&mut seed) as usize % 64;
                let ob = lcg(&mut seed) as usize % 64;
                let (pa, pb) = (&a[oa..], &b[ob..]);
                assert_eq!((d.sad)(pa, sa, pb, sb, w, h), (s.sad)(pa, sa, pb, sb, w, h), "sad {w}x{h} round {round}");
                assert_eq!((d.ssd)(pa, sa, pb, sb, w, h), (s.ssd)(pa, sa, pb, sb, w, h), "ssd {w}x{h} round {round}");
                assert_eq!((d.satd)(pa, sa, pb, sb, w, h), (s.satd)(pa, sa, pb, sb, w, h), "satd {w}x{h} round {round}");
            }
        }
    }

    #[test]
    fn full_deflection_is_exact() {
        let a = vec![0u8; 64 * 64];
        let b = vec![255u8; 64 * 64];
        let mut d = DistortionDsp::<u8>::scalar();
        install(&mut d, Cpu { neon: true, ..Cpu::SCALAR });
        assert_eq!((d.sad)(&a, 64, &b, 64, 64, 64), 255 * 4096);
        assert_eq!((d.ssd)(&a, 64, &b, 64, 64, 64), 255u64 * 255 * 4096);
        assert_eq!((d.satd)(&a, 64, &b, 64, 64, 64), 256 * ((16 * 255 + 1) >> 1));
    }
}
