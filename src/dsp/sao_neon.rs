//! NEON version of the H.265 encoder's SAO edge statistics
//! ([`super::distortion::SaoEdgeStatsFn`]), for both sample widths: the
//! same arithmetic as `sao_x86`, eight samples a vector as i16 lanes.
//!
//! `sign(r - a)` is the difference of the two compare masks, the category
//! `2 + sign(r - a) + sign(r - b)`, and per category an equality mask
//! subtracted into a lane counter and the masked error widened and added
//! into i32 lane sums. A sample near a rail (`reach` of 0 or of `max`) is
//! also reported one by one, as the scalar reference reports it.
//!
//! Every sum is an integer sum, so the order the lanes add in cannot change
//! the result: the statistics are the scalar reference's exactly. A `max`
//! above 32767 (samples an i16 compare would read as negative) is the
//! reference's.

#![cfg(target_arch = "aarch64")]

use std::arch::aarch64::*;

use super::Cpu;
use super::distortion::{DistortionDsp, sao_edge_stats_scalar};
use crate::sample::Sample;

/// Eight samples at `p` as eight i16.
#[inline(always)]
unsafe fn load8<S: Sample>(p: *const S) -> int16x8_t {
    unsafe {
        if S::BYTES == 1 {
            vreinterpretq_s16_u16(vmovl_u8(vld1_u8(p as *const u8)))
        } else {
            vld1q_s16(p as *const i16)
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn sao_edge_stats<S: Sample>(
    rec: &[S],
    origin: usize,
    stride: usize,
    src: &[S],
    src_stride: usize,
    w: usize,
    h: usize,
    na: isize,
    nb: isize,
    max: i32,
    reach: i32,
    tally: &mut [[i64; 2]; 5],
    near: &mut Vec<(u8, u16, i32)>,
) {
    // The furthest either neighbour reaches, before and after the region,
    // and the source rows the region reads.
    let (lo, hi) = (na.min(nb).min(0), na.max(nb).max(0));
    let fits = h > 0
        && w > 0
        && origin as isize + lo >= 0
        && (origin + (h - 1) * stride + w) as isize + hi <= rec.len() as isize
        && (h - 1) * src_stride + w <= src.len();
    if !fits || max > 32767 || w < 8 {
        return sao_edge_stats_scalar(
            rec, origin, stride, src, src_stride, w, h, na, nb, max, reach, tally, near,
        );
    }
    unsafe {
        edge_impl(
            rec, origin, stride, src, src_stride, w, h, na, nb, max, reach, tally, near,
        )
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn edge_impl<S: Sample>(
    recs: &[S],
    origin: usize,
    stride: usize,
    srcs: &[S],
    src_stride: usize,
    w: usize,
    h: usize,
    na: isize,
    nb: isize,
    max: i32,
    reach: i32,
    tally: &mut [[i64; 2]; 5],
    near: &mut Vec<(u8, u16, i32)>,
) {
    unsafe {
        let (rec, src) = (recs.as_ptr(), srcs.as_ptr());
        let lo_rail = vdupq_n_s16(reach as i16);
        let hi_rail = vdupq_n_s16((max - reach) as i16);
        let two = vdupq_n_s16(2);
        let cats: [int16x8_t; 5] = std::array::from_fn(|c| vdupq_n_s16(c as i16));
        let mut totals = [[0i64; 2]; 5];
        for y in 0..h {
            let row = rec.add(origin + y * stride);
            let srow = src.add(y * src_stride);
            // Per-row lane counters and sums, folded into i64 at the end of
            // the row: a lane then holds at most w / 8 samples, well inside
            // i16 and i32 at any width.
            let mut cnt = [vdupq_n_s16(0); 5];
            let mut sums = [vdupq_n_s32(0); 5];
            let mut x = 0;
            while x + 8 <= w {
                let p = row.add(x);
                let r = load8(p);
                let a = load8(p.offset(na));
                let b = load8(p.offset(nb));
                let s = load8(srow.add(x));
                // sign(r - a) = (a > r mask) - (r > a mask), masks being -1.
                let sa = vsubq_s16(
                    vreinterpretq_s16_u16(vcgtq_s16(a, r)),
                    vreinterpretq_s16_u16(vcgtq_s16(r, a)),
                );
                let sb = vsubq_s16(
                    vreinterpretq_s16_u16(vcgtq_s16(b, r)),
                    vreinterpretq_s16_u16(vcgtq_s16(r, b)),
                );
                let e = vaddq_s16(vaddq_s16(sa, sb), two);
                let err = vsubq_s16(s, r);
                for c in 0..5 {
                    let m = vreinterpretq_s16_u16(vceqq_s16(e, cats[c]));
                    cnt[c] = vsubq_s16(cnt[c], m);
                    let me = vandq_s16(err, m);
                    sums[c] = vpadalq_s16(sums[c], me);
                }
                let rail = vorrq_u16(vcgtq_s16(lo_rail, r), vcgtq_s16(r, hi_rail));
                if vmaxvq_u16(rail) != 0 {
                    let (mut rv, mut ev, mut dv, mut mv) =
                        ([0i16; 8], [0i16; 8], [0i16; 8], [0u16; 8]);
                    vst1q_s16(rv.as_mut_ptr(), r);
                    vst1q_s16(ev.as_mut_ptr(), e);
                    vst1q_s16(dv.as_mut_ptr(), err);
                    vst1q_u16(mv.as_mut_ptr(), rail);
                    for l in 0..8 {
                        if mv[l] != 0 {
                            near.push((ev[l] as u8, rv[l] as u16, dv[l] as i32));
                        }
                    }
                }
                x += 8;
            }
            for c in 0..5 {
                totals[c][0] += i64::from(vaddlvq_s16(cnt[c]));
                totals[c][1] += vaddlvq_s32(sums[c]);
            }
            if x < w {
                // The last few samples of the row, as the reference does them.
                sao_edge_stats_scalar(
                    recs,
                    origin + y * stride + x,
                    stride,
                    &srcs[y * src_stride + x..],
                    src_stride,
                    w - x,
                    1,
                    na,
                    nb,
                    max,
                    reach,
                    tally,
                    near,
                );
            }
        }
        for c in 0..5 {
            tally[c][0] += totals[c][0];
            tally[c][1] += totals[c][1];
        }
    }
}

/// Install the NEON edge-statistics kernel when `cpu` has NEON.
pub fn install<S: Sample>(d: &mut DistortionDsp<S>, cpu: Cpu) {
    if cpu.neon {
        d.sao_edge_stats = sao_edge_stats::<S>;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> u32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*seed >> 33) as u32
    }

    /// The NEON kernel against the reference: the four classes' neighbour
    /// offsets, regions whose width is and is not a multiple of eight, and
    /// planes uniform over the depth, crushed towards either rail (so most
    /// samples come back as near), or flat (every category but 2 empty).
    fn sweep<S: Sample>(depths: &[u32]) {
        let s = DistortionDsp::<S>::scalar();
        let mut d = DistortionDsp::<S>::scalar();
        install(&mut d, Cpu::detect());
        let mut seed = 0x5a0e_u64;
        let stride = 80;
        for &bd in depths {
            let max = (1i32 << bd) - 1;
            for kind in 0..4 {
                let plane: Vec<S> = (0..stride * 80)
                    .map(|_| {
                        let v = match kind {
                            0 => lcg(&mut seed) as i32 % (max + 1),
                            1 => lcg(&mut seed) as i32 % 40,
                            2 => max - lcg(&mut seed) as i32 % 40,
                            _ => max / 2,
                        };
                        S::from_i32(v)
                    })
                    .collect();
                let src: Vec<S> = (0..stride * 80)
                    .map(|_| S::from_i32(lcg(&mut seed) as i32 % (max + 1)))
                    .collect();
                for &(dx, dy) in &[(1isize, 0isize), (0, 1), (1, 1), (-1, 1)] {
                    let (na, nb) = (-dy * stride as isize - dx, dy * stride as isize + dx);
                    for &(w, h) in &[
                        (8usize, 8usize),
                        (13, 7),
                        (32, 32),
                        (64, 64),
                        (5, 3),
                        (61, 2),
                    ] {
                        let origin = 2 * stride + 2;
                        let run = |t: &DistortionDsp<S>| {
                            let mut got = [[0i64; 2]; 5];
                            let mut near = Vec::new();
                            (t.sao_edge_stats)(
                                &plane,
                                origin,
                                stride,
                                &src[origin..],
                                stride,
                                w,
                                h,
                                na,
                                nb,
                                max,
                                31,
                                &mut got,
                                &mut near,
                            );
                            near.sort();
                            (got, near)
                        };
                        assert_eq!(
                            run(&d),
                            run(&s),
                            "{bd} bits, plane {kind}, class ({dx}, {dy}), {w}x{h}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn edge_stats_match_scalar_u8() {
        sweep::<u8>(&[8]);
    }

    #[test]
    fn edge_stats_match_scalar_u16() {
        sweep::<u16>(&[9, 10, 12, 14, 15]);
    }
}
