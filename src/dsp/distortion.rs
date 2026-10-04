//! Distortion metrics: how far one block of samples is from another.
//!
//! Shared by both encoders, because none of it is codec-specific — a sum of
//! absolute differences does not care which standard asked for it. These are
//! the kernels an encoder calls most: mode decision evaluates one per
//! candidate per block, and motion search evaluates one per position
//! searched, so between them they see more samples than everything else in
//! an encoder put together.
//!
//! Three metrics, and the choice between them is a real one:
//!
//! - **SAD**, the sum of absolute differences. Cheapest, and what a motion
//!   search uses for its wide passes.
//! - **SATD**, the same after a Hadamard transform. Costs a few times more
//!   and picks visibly better modes, because it measures the residual the
//!   way the transform that will actually code it does — a residual that
//!   happens to be a smooth ramp is expensive to code but cheap in SAD, and
//!   SATD is what notices.
//! - **SSD**, the sum of squared differences. What rate-distortion
//!   optimisation needs, since it is the distortion term that pairs with a
//!   bit count in a Lagrangian.
//!
//! All three take two strided views and a block size rather than a fixed
//! shape, so one entry serves every partition size in both codecs. If a
//! profile ever shows the size-generic dispatch costing more than it saves,
//! the table can grow per-size entries without any caller changing.

use super::Cpu;
use crate::sample::Sample;

/// Sum of absolute differences over a `w` by `h` block.
pub type SadFn<S> = fn(a: &[S], a_stride: usize, b: &[S], b_stride: usize, w: usize, h: usize) -> u32;
/// Sum of absolute Hadamard-transformed differences over a `w` by `h`
/// block, both multiples of four.
pub type SatdFn<S> = fn(a: &[S], a_stride: usize, b: &[S], b_stride: usize, w: usize, h: usize) -> u32;
/// Sum of squared differences over a `w` by `h` block. Wider than the
/// others because at 12 bits a 64x64 block overflows 32 bits.
pub type SsdFn<S> = fn(a: &[S], a_stride: usize, b: &[S], b_stride: usize, w: usize, h: usize) -> u64;

/// The encoder's SAO edge-offset statistics (its side of 8.7.3) over a `w
/// x h` region of `rec` starting at index `origin`, row stride `stride`,
/// every sample's two neighbours — at offsets `na` and `nb` — usable: each
/// sample's category `e = 2 + sign(r - a) + sign(r - b)` and its error
/// `src - r`, counted and summed per category into `tally[e] = [count,
/// sum]`. `src` starts at the region's first sample, stride `src_stride`.
/// A sample within `reach` of 0 or of `max` is also pushed onto `near` as
/// `(e, r, err)`: the SAO model treats those apart, since a clip at the
/// rail moves them by less than the offset.
pub type SaoEdgeStatsFn<S> = fn(
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
);

/// The integer moments of a weighted-prediction fit (the encoders'
/// `h265_wp`) over a `w x h` region: the sums of the reference samples `r`,
/// the source samples `c`, `r * r` and `r * c`, and the zero-motion SAD
/// `|c - r|`. Integer, so lane order cannot change them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WpMoments {
    /// Sum of `r`.
    pub sr: u64,
    /// Sum of `c`.
    pub sc: u64,
    /// Sum of `r * r`.
    pub srr: u64,
    /// Sum of `r * c`.
    pub src: u64,
    /// Sum of `|c - r|`.
    pub sad: u64,
}

/// [`WpMoments`] of the source `cur` (stride `cur_stride`) against the
/// reference `refp` (stride `ref_stride`), both starting at the region's
/// first sample.
pub type WpMomentsFn<S> = fn(cur: &[S], cur_stride: usize, refp: &[S], ref_stride: usize, w: usize, h: usize) -> WpMoments;

/// The zero-motion SAD of `cur` against `refp` weighted the way both
/// standards' explicit weighting predicts a whole-sample vector:
/// `|c - Clip(((r * weight + round) >> shift) + offset)|` summed, with
/// `round = 1 << (shift - 1)` (0 at a shift of 0) and the clip to
/// `0..=max`. `weight` is within -128..=255, `round` and `offset` are the
/// caller's (`offset` already scaled to the sample depth), `shift` is at
/// most 7.
pub type WeightedSadFn<S> = fn(cur: &[S], cur_stride: usize, refp: &[S], ref_stride: usize, w: usize, h: usize, weight: i32, shift: u32, offset: i32, max: i32) -> u64;

/// The distortion kernels, filled at run time from what the CPU has.
#[derive(Clone)]
pub struct DistortionDsp<S: Sample = u8> {
    /// Which CPU features the table was built for.
    pub cpu: Cpu,
    /// Sum of absolute differences.
    pub sad: SadFn<S>,
    /// Sum of absolute Hadamard-transformed differences.
    pub satd: SatdFn<S>,
    /// Sum of squared differences.
    pub ssd: SsdFn<S>,
    /// SAO edge-offset statistics.
    pub sao_edge_stats: SaoEdgeStatsFn<S>,
    /// A weighted-prediction fit's moments, one pass over a plane.
    pub wp_moments: WpMomentsFn<S>,
    /// A weighted-prediction fit's weighted zero-motion SAD.
    pub weighted_sad: WeightedSadFn<S>,
}

impl<S: Sample> DistortionDsp<S> {
    /// The scalar reference table — the executable definition every wider
    /// rung is checked against.
    pub fn scalar() -> Self {
        DistortionDsp {
            cpu: Cpu::SCALAR,
            sad: sad_scalar::<S>,
            satd: satd_scalar::<S>,
            ssd: ssd_scalar::<S>,
            sao_edge_stats: sao_edge_stats_scalar::<S>,
            wp_moments: wp_moments_scalar::<S>,
            weighted_sad: weighted_sad_scalar::<S>,
        }
    }

    /// The best table for `cpu`, built the way the decoders' tables are:
    /// the scalar reference first, then each rung of the ladder replacing
    /// the entries it has a kernel for.
    pub fn new(cpu: Cpu) -> Self {
        let mut d = Self::scalar();
        d.cpu = cpu;
        install_simd(&mut d, cpu);
        d
    }
}

/// The SIMD kernels, for 8-bit samples and for 16-bit ones (which the deep
/// encoders use at 9 to 14 bits; the 16-bit kernels are told no depth and
/// are exact for any `u16`). Dispatched on the sample type here rather than
/// through a method on [`Sample`], so this table's ladder does not touch the
/// trait the decoders share.
#[allow(unused_variables)]
fn install_simd<S: Sample>(d: &mut DistortionDsp<S>, cpu: Cpu) {
    use std::any::Any;
    if super::enc_simd_disabled("distortion") {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    super::sao_x86::install(d, cpu);
    let d = d as &mut dyn Any;
    if let Some(d) = d.downcast_mut::<DistortionDsp<u8>>() {
        #[cfg(target_arch = "x86_64")]
        super::distortion_x86::install(d, cpu);
        #[cfg(target_arch = "aarch64")]
        super::distortion_neon::install(d, cpu);
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        if cpu.simd128 {
            super::distortion_wasm128::install(d);
        }
    } else if let Some(d) = d.downcast_mut::<DistortionDsp<u16>>() {
        #[cfg(target_arch = "x86_64")]
        super::distortion_x86_u16::install(d, cpu);
        #[cfg(target_arch = "aarch64")]
        super::distortion_neon_u16::install(d, cpu);
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        if cpu.simd128 {
            super::distortion_wasm128_u16::install(d);
        }
    }
}

impl<S: Sample> Default for DistortionDsp<S> {
    fn default() -> Self {
        Self::scalar()
    }
}

pub(crate) fn wp_moments_scalar<S: Sample>(cur: &[S], cur_stride: usize, refp: &[S], ref_stride: usize, w: usize, h: usize) -> WpMoments {
    let mut m = WpMoments::default();
    for y in 0..h {
        let (rr, cr) = (&refp[y * ref_stride..][..w], &cur[y * cur_stride..][..w]);
        for (&r, &c) in rr.iter().zip(cr) {
            let (r, c) = (r.to_i32() as u64, c.to_i32() as u64);
            m.sr += r;
            m.sc += c;
            m.srr += r * r;
            m.src += r * c;
            m.sad += r.abs_diff(c);
        }
    }
    m
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn weighted_sad_scalar<S: Sample>(cur: &[S], cur_stride: usize, refp: &[S], ref_stride: usize, w: usize, h: usize, weight: i32, shift: u32, offset: i32, max: i32) -> u64 {
    let round = if shift >= 1 { 1 << (shift - 1) } else { 0 };
    let mut sad = 0u64;
    for y in 0..h {
        let (rr, cr) = (&refp[y * ref_stride..][..w], &cur[y * cur_stride..][..w]);
        for (&r, &c) in rr.iter().zip(cr) {
            let p = ((r.to_i32() * weight + round) >> shift) + offset;
            sad += u64::from(p.clamp(0, max).abs_diff(c.to_i32()));
        }
    }
    sad
}

pub(crate) fn sad_scalar<S: Sample>(a: &[S], a_stride: usize, b: &[S], b_stride: usize, w: usize, h: usize) -> u32 {
    let mut sum = 0u32;
    for y in 0..h {
        let (ra, rb) = (&a[y * a_stride..], &b[y * b_stride..]);
        for x in 0..w {
            sum += ra[x].to_i32().abs_diff(rb[x].to_i32());
        }
    }
    sum
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn sao_edge_stats_scalar<S: Sample>(
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
    for y in 0..h {
        for x in 0..w {
            let i = origin + y * stride + x;
            let r = rec[i].to_i32();
            let a = rec[(i as isize + na) as usize].to_i32();
            let b = rec[(i as isize + nb) as usize].to_i32();
            let e = (2 + (r - a).signum() + (r - b).signum()) as usize;
            let err = src[y * src_stride + x].to_i32() - r;
            tally[e][0] += 1;
            tally[e][1] += err as i64;
            if r < reach || max - r < reach {
                near.push((e as u8, r as u16, err));
            }
        }
    }
}

pub(crate) fn ssd_scalar<S: Sample>(a: &[S], a_stride: usize, b: &[S], b_stride: usize, w: usize, h: usize) -> u64 {
    let mut sum = 0u64;
    for y in 0..h {
        let (ra, rb) = (&a[y * a_stride..], &b[y * b_stride..]);
        for x in 0..w {
            let d = (ra[x].to_i32() - rb[x].to_i32()) as i64;
            sum += (d * d) as u64;
        }
    }
    sum
}

/// The 4x4 Hadamard butterfly, in place over rows then columns.
#[inline(always)]
fn hadamard4x4(d: &mut [i32; 16]) {
    for i in 0..4 {
        let (a, b, c, e) = (d[i * 4], d[i * 4 + 1], d[i * 4 + 2], d[i * 4 + 3]);
        let (s0, s1, s2, s3) = (a + e, b + c, b - c, a - e);
        d[i * 4] = s0 + s1;
        d[i * 4 + 1] = s3 + s2;
        d[i * 4 + 2] = s0 - s1;
        d[i * 4 + 3] = s3 - s2;
    }
    for j in 0..4 {
        let (a, b, c, e) = (d[j], d[4 + j], d[8 + j], d[12 + j]);
        let (s0, s1, s2, s3) = (a + e, b + c, b - c, a - e);
        d[j] = s0 + s1;
        d[4 + j] = s3 + s2;
        d[8 + j] = s0 - s1;
        d[12 + j] = s3 - s2;
    }
}

/// SATD over 4x4 tiles. The `(sum + 1) >> 1` is the normalisation every
/// encoder since JM uses, chosen so that a SATD is on roughly the same
/// scale as the SAD of the same block and the two can share a Lagrangian
/// constant; it matters only that it is consistent, and it is stated here
/// so nobody has to infer it from a magic number later.
pub(crate) fn satd_scalar<S: Sample>(a: &[S], a_stride: usize, b: &[S], b_stride: usize, w: usize, h: usize) -> u32 {
    debug_assert!(w % 4 == 0 && h % 4 == 0, "SATD wants a multiple of four");
    let mut total = 0u32;
    for by in (0..h).step_by(4) {
        for bx in (0..w).step_by(4) {
            let mut d = [0i32; 16];
            for y in 0..4 {
                let (ra, rb) = (&a[(by + y) * a_stride + bx..], &b[(by + y) * b_stride + bx..]);
                for x in 0..4 {
                    d[y * 4 + x] = ra[x].to_i32() - rb[x].to_i32();
                }
            }
            hadamard4x4(&mut d);
            let sum: u32 = d.iter().map(|v| v.unsigned_abs()).sum();
            total += (sum + 1) >> 1;
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The weighted-prediction kernels of the host's table against the
    /// scalar references, 8- and 16-bit: random planes, rails, every weight
    /// either standard's table carries at every denominator, offsets at
    /// both ends, odd widths that leave a tail, and strides wider than the
    /// region. Holds on every architecture CI runs (AVX2 on x86-64, NEON on
    /// AArch64).
    #[test]
    fn weighted_prediction_kernels_match_scalar() {
        fn run<S: Sample>(max: i32) {
            let d = DistortionDsp::<S>::new(Cpu::detect());
            let s = DistortionDsp::<S>::scalar();
            let mut seed = 0x77u64;
            let mut checked = 0;
            for trial in 0..4 {
                for &(w, h) in &[(1usize, 1usize), (7, 3), (16, 2), (31, 5), (32, 4), (33, 3), (64, 9), (100, 7), (200, 3)] {
                    let stride = w + 13;
                    let plane = |seed: &mut u64| -> Vec<S> {
                        (0..stride * h)
                            .map(|_| {
                                let v = match trial {
                                    0 | 1 => (lcg(seed) % (max as u64 + 1)) as i32,
                                    2 => [0, max][(lcg(seed) & 1) as usize],
                                    _ => max,
                                };
                                S::from_i32(v)
                            })
                            .collect()
                    };
                    let (cur, refp) = (plane(&mut seed), plane(&mut seed));
                    assert_eq!((d.wp_moments)(&cur, stride, &refp, stride, w, h), (s.wp_moments)(&cur, stride, &refp, stride, w, h), "moments {w}x{h} trial {trial}");
                    let depth_scale = if max > 255 { 4 } else { 1 };
                    for weight in [-128, -77, -1, 0, 1, 31, 32, 63, 64, 65, 127, 191, 255] {
                        for shift in 0..=7u32 {
                            for offset in [-128 * depth_scale, -3, 0, 5, 127 * depth_scale] {
                                let want = (s.weighted_sad)(&cur, stride, &refp, stride, w, h, weight, shift, offset, max);
                                let got = (d.weighted_sad)(&cur, stride, &refp, stride, w, h, weight, shift, offset, max);
                                assert_eq!(got, want, "weighted_sad {w}x{h} weight {weight} shift {shift} offset {offset} trial {trial}");
                                checked += 1;
                            }
                        }
                    }
                }
            }
            assert!(checked > 0);
        }
        run::<u8>(255);
        run::<u16>(1023);
        run::<u16>(4095);
    }

    fn lcg(s: &mut u64) -> u64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *s >> 33
    }

    /// A block against itself is zero distortion by every metric, which is
    /// the one value all three must agree on.
    #[test]
    fn a_block_against_itself_is_zero() {
        let mut seed = 5u64;
        let p: Vec<u8> = (0..64 * 64).map(|_| lcg(&mut seed) as u8).collect();
        for &(w, h) in &[(4, 4), (8, 8), (16, 16), (16, 8), (8, 16), (64, 64)] {
            assert_eq!(sad_scalar(&p, 64, &p, 64, w, h), 0);
            assert_eq!(ssd_scalar(&p, 64, &p, 64, w, h), 0);
            assert_eq!(satd_scalar(&p, 64, &p, 64, w, h), 0);
        }
    }

    /// A constant offset has a closed form for each metric: SAD is the
    /// offset times the area, SSD its square times the area, and SATD sees
    /// it as pure DC — one coefficient of 16d a tile, halved.
    #[test]
    fn a_constant_offset_has_a_closed_form() {
        for d in [1i32, 7, 30] {
            let a = vec![100u8; 64 * 64];
            let b = vec![(100 + d) as u8; 64 * 64];
            for &(w, h) in &[(4, 4), (8, 8), (16, 16), (16, 4)] {
                let area = (w * h) as u32;
                assert_eq!(sad_scalar(&a, 64, &b, 64, w, h), d as u32 * area);
                assert_eq!(ssd_scalar(&a, 64, &b, 64, w, h), (d * d) as u64 * area as u64);
                let tiles = area / 16;
                assert_eq!(satd_scalar(&a, 64, &b, 64, w, h), tiles * ((16 * d as u32 + 1) >> 1));
            }
        }
    }

    /// SATD is the sum of absolute Hadamard coefficients, so it must agree
    /// with the transform computed the long way — a matrix multiply by the
    /// 4x4 Hadamard matrix, which shares no code with the butterfly.
    #[test]
    fn satd_agrees_with_a_direct_hadamard() {
        const H: [[i32; 4]; 4] = [[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, -1, 1], [1, -1, 1, -1]];
        let mut seed = 99u64;
        for _ in 0..200 {
            let a: Vec<u8> = (0..16).map(|_| lcg(&mut seed) as u8).collect();
            let b: Vec<u8> = (0..16).map(|_| lcg(&mut seed) as u8).collect();
            let mut diff = [[0i32; 4]; 4];
            for y in 0..4 {
                for x in 0..4 {
                    diff[y][x] = a[y * 4 + x] as i32 - b[y * 4 + x] as i32;
                }
            }
            // H * diff * H^T, term by term.
            let mut want = 0u32;
            for u in 0..4 {
                for v in 0..4 {
                    let mut acc = 0i32;
                    for y in 0..4 {
                        for x in 0..4 {
                            acc += H[u][y] * diff[y][x] * H[v][x];
                        }
                    }
                    want += acc.unsigned_abs();
                }
            }
            assert_eq!(satd_scalar(&a, 4, &b, 4, 4, 4), (want + 1) >> 1);
        }
    }

    /// Strides must be honoured: the same block read out of a wider plane
    /// gives the same answer.
    #[test]
    fn strides_are_honoured() {
        let mut seed = 21u64;
        let wide: Vec<u8> = (0..64 * 64).map(|_| lcg(&mut seed) as u8).collect();
        let other: Vec<u8> = (0..64 * 64).map(|_| lcg(&mut seed) as u8).collect();
        let mut packed_a = Vec::new();
        let mut packed_b = Vec::new();
        for y in 0..8 {
            packed_a.extend_from_slice(&wide[(3 + y) * 64 + 5..(3 + y) * 64 + 13]);
            packed_b.extend_from_slice(&other[(3 + y) * 64 + 5..(3 + y) * 64 + 13]);
        }
        let a = &wide[3 * 64 + 5..];
        let b = &other[3 * 64 + 5..];
        assert_eq!(sad_scalar(a, 64, b, 64, 8, 8), sad_scalar(&packed_a, 8, &packed_b, 8, 8, 8));
        assert_eq!(ssd_scalar(a, 64, b, 64, 8, 8), ssd_scalar(&packed_a, 8, &packed_b, 8, 8, 8));
        assert_eq!(satd_scalar(a, 64, b, 64, 8, 8), satd_scalar(&packed_a, 8, &packed_b, 8, 8, 8));
    }

    /// Ten bits per sample must not overflow, and a 64x64 SSD at full
    /// deflection is why `ssd` returns 64 bits: it is 2^38 there.
    #[test]
    fn wide_samples_do_not_overflow() {
        let a = vec![0u16; 64 * 64];
        let b = vec![1023u16; 64 * 64];
        assert_eq!(sad_scalar(&a, 64, &b, 64, 64, 64), 1023 * 4096);
        assert_eq!(ssd_scalar(&a, 64, &b, 64, 64, 64), 1023u64 * 1023 * 4096);
        assert_eq!(satd_scalar(&a, 64, &b, 64, 64, 64), 256 * ((16 * 1023 + 1) >> 1));
    }
}
