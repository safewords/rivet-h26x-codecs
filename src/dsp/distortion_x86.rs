//! x86-64 SIMD versions of the distortion metrics, for 8-bit samples.
//!
//! These are the kernels the encoders call most — a profile of either
//! encoder puts `satd` first at 12–49% of self time and `sad` at 5–11% in
//! every inter configuration (docs/encode_speed.md) — and they are the
//! easiest kind of kernel to vectorise exactly, because there is no
//! rounding anywhere in them: SAD and SSD are sums of integers, and SATD's
//! one rounding step, `(tile + 1) >> 1`, is applied per 4x4 tile *after*
//! the tile's sum is complete, which is how the scalar reference does it
//! and how these do it.
//!
//! Written once and compiled per rung by the `kernels!` macro, as the
//! decoder's kernels are. SSE2 carries everything — `psadbw` is the SAD,
//! `pmaddwd` against itself is the SSD, and the Hadamard is adds, subtracts
//! and a 4x4 transpose of 16-bit lanes; SSSE3 replaces the SATD's two-op
//! absolute value with `pabsw`; AVX re-encodes SSE4.1 for VEX. AVX2 then
//! takes the shapes sixteen or more samples wide, which are four Hadamard
//! tiles or a whole row per vector.
//!
//! The SATD's lane layout is worth stating because the transpose depends
//! on it. A vector of eight i16 holds one row of two horizontally adjacent
//! 4x4 tiles, `[A0 A1 A2 A3 B0 B1 B2 B3]`; four such vectors are the four
//! rows. The vertical Hadamard is then lane-wise across the four vectors,
//! the transpose turns rows into columns *within each tile* (three levels
//! of unpack, which never cross a 64-bit half), and the same four
//! butterflies run again. Widths of four are handled the same way with two
//! *vertically* adjacent tiles in the two halves, and a lone 4x4 tile
//! rides in the low half with zeros above it, which contribute exactly
//! zero after rounding. Intermediates never leave i16: a difference is at
//! most 255, after one Hadamard stage at most 1020, after both at most
//! 4080, and a tile's sixteen absolute values sum to at most 65280 — which
//! is why the per-lane sums are widened to i32 by `pmaddwd` before the
//! two halves of a tile are added.

#![cfg(target_arch = "x86_64")]

use super::Cpu;
use super::distortion::DistortionDsp;

macro_rules! kernels {
    ($feat:literal, $lvl:tt) => {
        use std::arch::x86_64::*;

        use crate::dsp::distortion::{DistortionDsp, sad_scalar, satd_scalar, ssd_scalar};

        crate::dsp::x86_compat::compat_core!($feat, $lvl);

        /// Every kernel.
        pub(crate) fn install_all(d: &mut DistortionDsp<u8>) {
            d.sad = sad;
            d.satd = satd;
            d.ssd = ssd;
        }

        /// The SATD alone — what a rung whose only change is a better
        /// absolute value re-installs.
        pub(crate) fn install_satd(d: &mut DistortionDsp<u8>) {
            d.satd = satd;
        }

        /// Four bytes at `p` in the low lane of a vector, the rest zero.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn load4(p: *const u8) -> __m128i {
            unsafe { _mm_cvtsi32_si128((p as *const u32).read_unaligned() as i32) }
        }

        /// Eight bytes at `p` in the low half of a vector, the rest zero.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn load8(p: *const u8) -> __m128i {
            unsafe { _mm_loadl_epi64(p as *const __m128i) }
        }

        /// The two 64-bit lanes of a `psadbw` accumulator, summed.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn sad_lanes(acc: __m128i) -> u32 {
            (_mm_cvtsi128_si32(acc) as u32)
                .wrapping_add(_mm_cvtsi128_si32(_mm_srli_si128(acc, 8)) as u32)
        }

        // ------------------------------------------------------------------
        // SAD
        // ------------------------------------------------------------------

        #[target_feature(enable = $feat)]
        unsafe fn sad_impl(
            a: *const u8,
            sa: usize,
            b: *const u8,
            sb: usize,
            w: usize,
            h: usize,
        ) -> u32 {
            unsafe {
                let mut acc = _mm_setzero_si128();
                for y in 0..h {
                    let ra = a.add(y * sa);
                    let rb = b.add(y * sb);
                    let mut x = 0;
                    while x + 16 <= w {
                        let va = _mm_loadu_si128(ra.add(x) as *const __m128i);
                        let vb = _mm_loadu_si128(rb.add(x) as *const __m128i);
                        acc = _mm_add_epi64(acc, _mm_sad_epu8(va, vb));
                        x += 16;
                    }
                    if x + 8 <= w {
                        acc = _mm_add_epi64(acc, _mm_sad_epu8(load8(ra.add(x)), load8(rb.add(x))));
                        x += 8;
                    }
                    if x + 4 <= w {
                        acc = _mm_add_epi64(acc, _mm_sad_epu8(load4(ra.add(x)), load4(rb.add(x))));
                    }
                }
                sad_lanes(acc)
            }
        }

        pub(crate) fn sad(
            a: &[u8],
            a_stride: usize,
            b: &[u8],
            b_stride: usize,
            w: usize,
            h: usize,
        ) -> u32 {
            if w % 4 != 0 || h == 0 {
                return sad_scalar(a, a_stride, b, b_stride, w, h);
            }
            // The scalar reference indexes and would panic; this reads
            // through pointers, so the bound is checked once here.
            assert!(
                a.len() >= (h - 1) * a_stride + w && b.len() >= (h - 1) * b_stride + w,
                "block out of range"
            );
            unsafe { sad_impl(a.as_ptr(), a_stride, b.as_ptr(), b_stride, w, h) }
        }

        // ------------------------------------------------------------------
        // SSD
        // ------------------------------------------------------------------

        /// Sum of squared differences of the eight byte pairs in the low
        /// halves of `va` and `vb`, as four i32 (pairs of lanes summed).
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn sq8(va: __m128i, vb: __m128i) -> __m128i {
            unsafe {
                let d = _mm_sub_epi16(zx8(va), zx8(vb));
                _mm_madd_epi16(d, d)
            }
        }

        #[target_feature(enable = $feat)]
        unsafe fn ssd_impl(
            a: *const u8,
            sa: usize,
            b: *const u8,
            sb: usize,
            w: usize,
            h: usize,
        ) -> u64 {
            unsafe {
                let zero = _mm_setzero_si128();
                let mut acc = zero;
                for y in 0..h {
                    let ra = a.add(y * sa);
                    let rb = b.add(y * sb);
                    // One row's squares fit i32 lanes with room to spare
                    // (at most 4 * 130050 per lane for a 64-wide row);
                    // widening once a row keeps the 64-bit accumulator
                    // exact for any block.
                    let mut row = zero;
                    let mut x = 0;
                    while x + 16 <= w {
                        let va = _mm_loadu_si128(ra.add(x) as *const __m128i);
                        let vb = _mm_loadu_si128(rb.add(x) as *const __m128i);
                        let dh = _mm_sub_epi16(zx8h(va), zx8h(vb));
                        row = _mm_add_epi32(row, sq8(va, vb));
                        row = _mm_add_epi32(row, _mm_madd_epi16(dh, dh));
                        x += 16;
                    }
                    if x + 8 <= w {
                        row = _mm_add_epi32(row, sq8(load8(ra.add(x)), load8(rb.add(x))));
                        x += 8;
                    }
                    if x + 4 <= w {
                        row = _mm_add_epi32(row, sq8(load4(ra.add(x)), load4(rb.add(x))));
                    }
                    acc = _mm_add_epi64(acc, _mm_unpacklo_epi32(row, zero));
                    acc = _mm_add_epi64(acc, _mm_unpackhi_epi32(row, zero));
                }
                let mut out = [0u64; 2];
                _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, acc);
                out[0] + out[1]
            }
        }

        pub(crate) fn ssd(
            a: &[u8],
            a_stride: usize,
            b: &[u8],
            b_stride: usize,
            w: usize,
            h: usize,
        ) -> u64 {
            if w % 4 != 0 || h == 0 {
                return ssd_scalar(a, a_stride, b, b_stride, w, h);
            }
            assert!(
                a.len() >= (h - 1) * a_stride + w && b.len() >= (h - 1) * b_stride + w,
                "block out of range"
            );
            unsafe { ssd_impl(a.as_ptr(), a_stride, b.as_ptr(), b_stride, w, h) }
        }

        // ------------------------------------------------------------------
        // SATD
        // ------------------------------------------------------------------

        /// The 4-point Hadamard butterfly, lane-wise across four vectors.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn butterfly(r0: __m128i, r1: __m128i, r2: __m128i, r3: __m128i) -> [__m128i; 4] {
            let s0 = _mm_add_epi16(r0, r3);
            let s1 = _mm_add_epi16(r1, r2);
            let s2 = _mm_sub_epi16(r1, r2);
            let s3 = _mm_sub_epi16(r0, r3);
            [
                _mm_add_epi16(s0, s1),
                _mm_add_epi16(s3, s2),
                _mm_sub_epi16(s0, s1),
                _mm_sub_epi16(s3, s2),
            ]
        }

        /// SATD of the two 4x4 tiles held in `r0..r3` (one row each, tile
        /// A in the low half, B in the high), as `[A, A, B, B]` i32 with
        /// the per-tile `(sum + 1) >> 1` applied.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn satd_pair(r0: __m128i, r1: __m128i, r2: __m128i, r3: __m128i) -> __m128i {
            unsafe {
                let [t0, t1, t2, t3] = butterfly(r0, r1, r2, r3);
                // Transpose each tile: rows become columns, halves stay put.
                let u0 = _mm_unpacklo_epi16(t0, t1);
                let u1 = _mm_unpacklo_epi16(t2, t3);
                let u2 = _mm_unpackhi_epi16(t0, t1);
                let u3 = _mm_unpackhi_epi16(t2, t3);
                let v0 = _mm_unpacklo_epi32(u0, u1);
                let v1 = _mm_unpackhi_epi32(u0, u1);
                let v2 = _mm_unpacklo_epi32(u2, u3);
                let v3 = _mm_unpackhi_epi32(u2, u3);
                let c0 = _mm_unpacklo_epi64(v0, v2);
                let c1 = _mm_unpackhi_epi64(v0, v2);
                let c2 = _mm_unpacklo_epi64(v1, v3);
                let c3 = _mm_unpackhi_epi64(v1, v3);
                let [w0, w1, w2, w3] = butterfly(c0, c1, c2, c3);
                let s = _mm_add_epi16(
                    _mm_add_epi16(abs16(w0), abs16(w1)),
                    _mm_add_epi16(abs16(w2), abs16(w3)),
                );
                // [A01, A23, B01, B23] -> [A, A, B, B], then the tile rounding.
                let p = _mm_madd_epi16(s, _mm_set1_epi16(1));
                let q = _mm_add_epi32(p, _mm_shuffle_epi32(p, 0b10_11_00_01));
                _mm_srli_epi32(_mm_add_epi32(q, _mm_set1_epi32(1)), 1)
            }
        }

        /// Two tiles side by side: rows of eight bytes at `a` and `b`.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn tiles_h(a: *const u8, sa: usize, b: *const u8, sb: usize) -> __m128i {
            unsafe {
                let row =
                    |y: usize| _mm_sub_epi16(zx8(load8(a.add(y * sa))), zx8(load8(b.add(y * sb))));
                satd_pair(row(0), row(1), row(2), row(3))
            }
        }

        /// Two tiles one above the other: rows `y` and `y + 4` of a
        /// four-wide block share a vector.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn tiles_v(a: *const u8, sa: usize, b: *const u8, sb: usize) -> __m128i {
            unsafe {
                let row = |y: usize| {
                    let va = _mm_unpacklo_epi32(load4(a.add(y * sa)), load4(a.add((y + 4) * sa)));
                    let vb = _mm_unpacklo_epi32(load4(b.add(y * sb)), load4(b.add((y + 4) * sb)));
                    _mm_sub_epi16(zx8(va), zx8(vb))
                };
                satd_pair(row(0), row(1), row(2), row(3))
            }
        }

        /// One tile, in the low half; the zero tile above it costs nothing.
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn tile_1(a: *const u8, sa: usize, b: *const u8, sb: usize) -> __m128i {
            unsafe {
                let row =
                    |y: usize| _mm_sub_epi16(zx8(load4(a.add(y * sa))), zx8(load4(b.add(y * sb))));
                satd_pair(row(0), row(1), row(2), row(3))
            }
        }

        #[target_feature(enable = $feat)]
        unsafe fn satd_impl(
            a: *const u8,
            sa: usize,
            b: *const u8,
            sb: usize,
            w: usize,
            h: usize,
        ) -> u32 {
            unsafe {
                let mut acc = _mm_setzero_si128();
                if w == 4 {
                    let mut y = 0;
                    while y + 8 <= h {
                        acc = _mm_add_epi32(acc, tiles_v(a.add(y * sa), sa, b.add(y * sb), sb));
                        y += 8;
                    }
                    if y < h {
                        acc = _mm_add_epi32(acc, tile_1(a.add(y * sa), sa, b.add(y * sb), sb));
                    }
                } else {
                    let mut y = 0;
                    while y < h {
                        let ra = a.add(y * sa);
                        let rb = b.add(y * sb);
                        let mut x = 0;
                        while x + 8 <= w {
                            acc = _mm_add_epi32(acc, tiles_h(ra.add(x), sa, rb.add(x), sb));
                            x += 8;
                        }
                        if x < w {
                            acc = _mm_add_epi32(acc, tile_1(ra.add(x), sa, rb.add(x), sb));
                        }
                        y += 4;
                    }
                }
                // Lanes are [A, A, B, B] sums: one of each.
                (_mm_cvtsi128_si32(acc) as u32)
                    .wrapping_add(_mm_cvtsi128_si32(_mm_srli_si128(acc, 8)) as u32)
            }
        }

        pub(crate) fn satd(
            a: &[u8],
            a_stride: usize,
            b: &[u8],
            b_stride: usize,
            w: usize,
            h: usize,
        ) -> u32 {
            if w % 4 != 0 || h % 4 != 0 || h == 0 {
                return satd_scalar(a, a_stride, b, b_stride, w, h);
            }
            assert!(
                a.len() >= (h - 1) * a_stride + w && b.len() >= (h - 1) * b_stride + w,
                "block out of range"
            );
            unsafe { satd_impl(a.as_ptr(), a_stride, b.as_ptr(), b_stride, w, h) }
        }
    };
}

/// SSE2: baseline on x86-64, so this rung is the one that makes the scalar
/// kernels unreachable on this architecture.
pub(crate) mod sse2 {
    #![allow(dead_code)]
    kernels!("sse2", sse2);
}

/// SSSE3: `pabsw` in the SATD.
pub(crate) mod ssse3 {
    #![allow(dead_code)]
    kernels!("ssse3", ssse3);
}

/// AVX: the SSE4.1 primitive set, VEX-encoded.
pub(crate) mod avx {
    #![allow(dead_code)]
    kernels!("avx", sse41);
}

/// AVX2: sixteen samples a vector, for the block widths that have them,
/// and the SATD of eight-wide blocks in whole 8x8s (two rows a vector,
/// four tiles a transform). Narrower blocks take the [`avx`] kernels — the
/// 256-bit body would spend half its lanes on nothing.
pub(crate) mod avx2 {
    use std::arch::x86_64::*;

    use crate::dsp::distortion::{DistortionDsp, WpMoments};

    pub(crate) fn install(d: &mut DistortionDsp<u8>) {
        d.sad = sad;
        d.satd = satd;
        d.ssd = ssd;
        d.wp_moments = wp_moments;
        d.weighted_sad = weighted_sad;
    }

    // ------------------------------------------------------------------
    // Weighted-prediction fit: moments and weighted SAD (8-bit)
    // ------------------------------------------------------------------

    /// Four u64 lanes summed.
    #[target_feature(enable = "avx2")]
    #[inline]
    fn hsum_u64(v: __m256i) -> u64 {
        let s = _mm_add_epi64(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
        (_mm_cvtsi128_si64(s) as u64).wrapping_add(_mm_extract_epi64::<1>(s) as u64)
    }

    /// Eight i32 lanes, each non-negative, widened and added into four
    /// u64 lanes.
    #[target_feature(enable = "avx2")]
    #[inline]
    fn widen_add(acc: __m256i, v: __m256i) -> __m256i {
        let z = _mm256_setzero_si256();
        _mm256_add_epi64(
            acc,
            _mm256_add_epi64(_mm256_unpacklo_epi32(v, z), _mm256_unpackhi_epi32(v, z)),
        )
    }

    pub(crate) fn wp_moments(
        cur: &[u8],
        cur_stride: usize,
        refp: &[u8],
        ref_stride: usize,
        w: usize,
        h: usize,
    ) -> WpMoments {
        if w == 0 || h == 0 {
            return WpMoments::default();
        }
        assert!(
            cur.len() >= (h - 1) * cur_stride + w && refp.len() >= (h - 1) * ref_stride + w,
            "region out of range"
        );
        // SAFETY: `install` only installs this for a CPU with AVX2, and
        // every row's loads are inside the `w` samples checked above.
        unsafe { wp_moments_impl(cur.as_ptr(), cur_stride, refp.as_ptr(), ref_stride, w, h) }
    }

    /// 32 samples a step: `psadbw` against zero sums `r` and `c`, against
    /// each other the SAD; `pmaddwd` of the zero-extended words gives
    /// `r * r` and `r * c` in pairs (each pair at most 2 * 255^2). Those
    /// i32 lanes are widened into u64 once a row: a row of up to 2^14
    /// samples puts at most 2^9 pairs, under 2^26, in a lane.
    #[target_feature(enable = "avx2")]
    unsafe fn wp_moments_impl(
        cur: *const u8,
        cur_stride: usize,
        refp: *const u8,
        ref_stride: usize,
        w: usize,
        h: usize,
    ) -> WpMoments {
        unsafe {
            let z = _mm256_setzero_si256();
            let (mut sr, mut sc, mut sad, mut srr, mut src) = (z, z, z, z, z);
            let mut m = WpMoments::default();
            let wide = if w <= 1 << 14 { w & !31 } else { 0 };
            for y in 0..h {
                let (c, r) = (cur.add(y * cur_stride), refp.add(y * ref_stride));
                let (mut rr, mut rc) = (z, z);
                let mut x = 0;
                while x < wide {
                    let vc = _mm256_loadu_si256(c.add(x) as *const __m256i);
                    let vr = _mm256_loadu_si256(r.add(x) as *const __m256i);
                    sr = _mm256_add_epi64(sr, _mm256_sad_epu8(vr, z));
                    sc = _mm256_add_epi64(sc, _mm256_sad_epu8(vc, z));
                    sad = _mm256_add_epi64(sad, _mm256_sad_epu8(vr, vc));
                    let (rl, rh) = (_mm256_unpacklo_epi8(vr, z), _mm256_unpackhi_epi8(vr, z));
                    let (cl, ch) = (_mm256_unpacklo_epi8(vc, z), _mm256_unpackhi_epi8(vc, z));
                    rr = _mm256_add_epi32(
                        rr,
                        _mm256_add_epi32(_mm256_madd_epi16(rl, rl), _mm256_madd_epi16(rh, rh)),
                    );
                    rc = _mm256_add_epi32(
                        rc,
                        _mm256_add_epi32(_mm256_madd_epi16(rl, cl), _mm256_madd_epi16(rh, ch)),
                    );
                    x += 32;
                }
                srr = widen_add(srr, rr);
                src = widen_add(src, rc);
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
            m.sr += hsum_u64(sr);
            m.sc += hsum_u64(sc);
            m.sad += hsum_u64(sad);
            m.srr += hsum_u64(srr);
            m.src += hsum_u64(src);
            m
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn weighted_sad(
        cur: &[u8],
        cur_stride: usize,
        refp: &[u8],
        ref_stride: usize,
        w: usize,
        h: usize,
        weight: i32,
        shift: u32,
        offset: i32,
        max: i32,
    ) -> u64 {
        if w == 0 || h == 0 {
            return 0;
        }
        assert!(
            cur.len() >= (h - 1) * cur_stride + w && refp.len() >= (h - 1) * ref_stride + w,
            "region out of range"
        );
        // The word pairs below hold the weight and the rounding term as
        // i16, and the clip assumes an 8-bit ceiling; anything else (no
        // caller passes it) takes the reference.
        if !(-32768..=32767).contains(&weight) || shift > 14 || max != 255 {
            return crate::dsp::distortion::weighted_sad_scalar(
                cur, cur_stride, refp, ref_stride, w, h, weight, shift, offset, max,
            );
        }
        // SAFETY: AVX2 as above; the loads stay inside the checked rows.
        unsafe {
            weighted_sad_impl(
                cur.as_ptr(),
                cur_stride,
                refp.as_ptr(),
                ref_stride,
                w,
                h,
                weight,
                shift,
                offset,
            )
        }
    }

    /// 16 samples a step: the reference words interleaved with ones, so one
    /// `pmaddwd` against `(weight, round)` is `r * weight + round` in i32;
    /// then the arithmetic shift, the offset, the clip to 0..=255 and the
    /// absolute difference, all in i32 as the reference has them. The sums
    /// (at most 255 a sample) are widened into u64 once a row.
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx2")]
    unsafe fn weighted_sad_impl(
        cur: *const u8,
        cur_stride: usize,
        refp: *const u8,
        ref_stride: usize,
        w: usize,
        h: usize,
        weight: i32,
        shift: u32,
        offset: i32,
    ) -> u64 {
        unsafe {
            let round = if shift >= 1 { 1i32 << (shift - 1) } else { 0 };
            let wr = _mm256_set1_epi32((weight & 0xffff) | (round << 16));
            let ones = _mm256_set1_epi16(1);
            let sh = _mm_cvtsi32_si128(shift as i32);
            let off = _mm256_set1_epi32(offset);
            let (lo, hi) = (_mm256_setzero_si256(), _mm256_set1_epi32(255));
            let mut acc = _mm256_setzero_si256();
            let mut total = 0u64;
            let wide = if w <= 1 << 20 { w & !15 } else { 0 };
            let predict = |r: __m256i| {
                _mm256_min_epi32(
                    _mm256_max_epi32(
                        _mm256_add_epi32(_mm256_sra_epi32(_mm256_madd_epi16(r, wr), sh), off),
                        lo,
                    ),
                    hi,
                )
            };
            for y in 0..h {
                let (c, r) = (cur.add(y * cur_stride), refp.add(y * ref_stride));
                let mut row = _mm256_setzero_si256();
                let mut x = 0;
                while x < wide {
                    // Sixteen samples as words, in the lane order
                    // `cvtepu8` leaves them: 0-7 low, 8-15 high.
                    let vr = _mm256_cvtepu8_epi16(_mm_loadu_si128(r.add(x) as *const __m128i));
                    let vc = _mm256_cvtepu8_epi16(_mm_loadu_si128(c.add(x) as *const __m128i));
                    let (pa, pb) = (
                        predict(_mm256_unpacklo_epi16(vr, ones)),
                        predict(_mm256_unpackhi_epi16(vr, ones)),
                    );
                    let (ca, cb) = (_mm256_unpacklo_epi16(vc, lo), _mm256_unpackhi_epi16(vc, lo));
                    row = _mm256_add_epi32(
                        row,
                        _mm256_add_epi32(
                            _mm256_abs_epi32(_mm256_sub_epi32(pa, ca)),
                            _mm256_abs_epi32(_mm256_sub_epi32(pb, cb)),
                        ),
                    );
                    x += 16;
                }
                acc = widen_add(acc, row);
                while x < w {
                    let p = ((i32::from(*r.add(x)) * weight + round) >> shift) + offset;
                    total += u64::from(p.clamp(0, 255).abs_diff(i32::from(*c.add(x))));
                    x += 1;
                }
            }
            total + hsum_u64(acc)
        }
    }

    /// Sixteen bytes at `p`, zero-extended to sixteen i16.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn load16w(p: *const u8) -> __m256i {
        unsafe { _mm256_cvtepu8_epi16(_mm_loadu_si128(p as *const __m128i)) }
    }

    /// Two rows of sixteen bytes, one per 128-bit lane.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn load2x16(p: *const u8, stride: usize) -> __m256i {
        unsafe {
            let lo = _mm_loadu_si128(p as *const __m128i);
            let hi = _mm_loadu_si128(p.add(stride) as *const __m128i);
            _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1)
        }
    }

    /// The four 64-bit lanes of a `vpsadbw` accumulator, summed.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn sum64(acc: __m256i) -> u64 {
        unsafe {
            let mut out = [0u64; 4];
            _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, acc);
            out[0] + out[1] + out[2] + out[3]
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn sad_impl(
        a: *const u8,
        sa: usize,
        b: *const u8,
        sb: usize,
        w: usize,
        h: usize,
    ) -> u32 {
        unsafe {
            let mut acc = _mm256_setzero_si256();
            if w == 16 {
                let mut y = 0;
                while y + 2 <= h {
                    let va = load2x16(a.add(y * sa), sa);
                    let vb = load2x16(b.add(y * sb), sb);
                    acc = _mm256_add_epi64(acc, _mm256_sad_epu8(va, vb));
                    y += 2;
                }
                if y < h {
                    let va =
                        _mm256_castsi128_si256(_mm_loadu_si128(a.add(y * sa) as *const __m128i));
                    let vb =
                        _mm256_castsi128_si256(_mm_loadu_si128(b.add(y * sb) as *const __m128i));
                    acc = _mm256_add_epi64(acc, _mm256_sad_epu8(va, vb));
                }
            } else {
                for y in 0..h {
                    let ra = a.add(y * sa);
                    let rb = b.add(y * sb);
                    let mut x = 0;
                    while x + 32 <= w {
                        let va = _mm256_loadu_si256(ra.add(x) as *const __m256i);
                        let vb = _mm256_loadu_si256(rb.add(x) as *const __m256i);
                        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(va, vb));
                        x += 32;
                    }
                    if x + 16 <= w {
                        let va =
                            _mm256_castsi128_si256(_mm_loadu_si128(ra.add(x) as *const __m128i));
                        let vb =
                            _mm256_castsi128_si256(_mm_loadu_si128(rb.add(x) as *const __m128i));
                        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(va, vb));
                    }
                }
            }
            sum64(acc) as u32
        }
    }

    fn sad(a: &[u8], a_stride: usize, b: &[u8], b_stride: usize, w: usize, h: usize) -> u32 {
        if !w.is_multiple_of(16) || h == 0 {
            return super::avx::sad(a, a_stride, b, b_stride, w, h);
        }
        assert!(
            a.len() >= (h - 1) * a_stride + w && b.len() >= (h - 1) * b_stride + w,
            "block out of range"
        );
        unsafe { sad_impl(a.as_ptr(), a_stride, b.as_ptr(), b_stride, w, h) }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn ssd_impl(
        a: *const u8,
        sa: usize,
        b: *const u8,
        sb: usize,
        w: usize,
        h: usize,
    ) -> u64 {
        unsafe {
            let zero = _mm256_setzero_si256();
            let mut acc = zero;
            for y in 0..h {
                let ra = a.add(y * sa);
                let rb = b.add(y * sb);
                let mut row = zero;
                let mut x = 0;
                while x < w {
                    let d = _mm256_sub_epi16(load16w(ra.add(x)), load16w(rb.add(x)));
                    row = _mm256_add_epi32(row, _mm256_madd_epi16(d, d));
                    x += 16;
                }
                acc = _mm256_add_epi64(acc, _mm256_unpacklo_epi32(row, zero));
                acc = _mm256_add_epi64(acc, _mm256_unpackhi_epi32(row, zero));
            }
            sum64(acc)
        }
    }

    fn ssd(a: &[u8], a_stride: usize, b: &[u8], b_stride: usize, w: usize, h: usize) -> u64 {
        if !w.is_multiple_of(16) || h == 0 {
            return super::avx::ssd(a, a_stride, b, b_stride, w, h);
        }
        assert!(
            a.len() >= (h - 1) * a_stride + w && b.len() >= (h - 1) * b_stride + w,
            "block out of range"
        );
        unsafe { ssd_impl(a.as_ptr(), a_stride, b.as_ptr(), b_stride, w, h) }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn butterfly(r0: __m256i, r1: __m256i, r2: __m256i, r3: __m256i) -> [__m256i; 4] {
        let s0 = _mm256_add_epi16(r0, r3);
        let s1 = _mm256_add_epi16(r1, r2);
        let s2 = _mm256_sub_epi16(r1, r2);
        let s3 = _mm256_sub_epi16(r0, r3);
        [
            _mm256_add_epi16(s0, s1),
            _mm256_add_epi16(s3, s2),
            _mm256_sub_epi16(s0, s1),
            _mm256_sub_epi16(s3, s2),
        ]
    }

    /// Four tiles, two per 128-bit lane, from four rows of sixteen i16
    /// differences: the 128-bit `satd_pair` twice over, since every unpack
    /// and shuffle here stays in its lane. Returns `[A, A, B, B, C, C, D,
    /// D]`, rounded per tile.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn satd_rows(d: [__m256i; 4]) -> __m256i {
        unsafe {
            let [t0, t1, t2, t3] = butterfly(d[0], d[1], d[2], d[3]);
            let u0 = _mm256_unpacklo_epi16(t0, t1);
            let u1 = _mm256_unpacklo_epi16(t2, t3);
            let u2 = _mm256_unpackhi_epi16(t0, t1);
            let u3 = _mm256_unpackhi_epi16(t2, t3);
            let v0 = _mm256_unpacklo_epi32(u0, u1);
            let v1 = _mm256_unpackhi_epi32(u0, u1);
            let v2 = _mm256_unpacklo_epi32(u2, u3);
            let v3 = _mm256_unpackhi_epi32(u2, u3);
            let c0 = _mm256_unpacklo_epi64(v0, v2);
            let c1 = _mm256_unpackhi_epi64(v0, v2);
            let c2 = _mm256_unpacklo_epi64(v1, v3);
            let c3 = _mm256_unpackhi_epi64(v1, v3);
            let [w0, w1, w2, w3] = butterfly(c0, c1, c2, c3);
            let s = _mm256_add_epi16(
                _mm256_add_epi16(_mm256_abs_epi16(w0), _mm256_abs_epi16(w1)),
                _mm256_add_epi16(_mm256_abs_epi16(w2), _mm256_abs_epi16(w3)),
            );
            let p = _mm256_madd_epi16(s, _mm256_set1_epi16(1));
            let q = _mm256_add_epi32(p, _mm256_shuffle_epi32(p, 0b10_11_00_01));
            _mm256_srli_epi32(_mm256_add_epi32(q, _mm256_set1_epi32(1)), 1)
        }
    }

    /// Four tiles across a sixteen-wide strip of four rows.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn satd_quad(a: *const u8, sa: usize, b: *const u8, sb: usize) -> __m256i {
        unsafe {
            let row = |y: usize| _mm256_sub_epi16(load16w(a.add(y * sa)), load16w(b.add(y * sb)));
            satd_rows([row(0), row(1), row(2), row(3)])
        }
    }

    /// The four tiles of an 8x8 block: rows 0..4 in the low lane, 4..8 in
    /// the high one, where the 128-bit kernel takes two pairs.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn satd_8x8(a: *const u8, sa: usize, b: *const u8, sb: usize) -> __m256i {
        unsafe {
            let two = |p: *const u8, s: usize, y: usize| {
                _mm256_cvtepu8_epi16(_mm_unpacklo_epi64(
                    _mm_loadl_epi64(p.add(y * s) as *const __m128i),
                    _mm_loadl_epi64(p.add((y + 4) * s) as *const __m128i),
                ))
            };
            let row = |y: usize| _mm256_sub_epi16(two(a, sa, y), two(b, sb, y));
            satd_rows([row(0), row(1), row(2), row(3)])
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn satd_impl(
        a: *const u8,
        sa: usize,
        b: *const u8,
        sb: usize,
        w: usize,
        h: usize,
    ) -> u32 {
        unsafe {
            let mut acc = _mm256_setzero_si256();
            let mut y = 0;
            if w == 8 {
                while y < h {
                    acc = _mm256_add_epi32(acc, satd_8x8(a.add(y * sa), sa, b.add(y * sb), sb));
                    y += 8;
                }
            }
            while y < h {
                let ra = a.add(y * sa);
                let rb = b.add(y * sb);
                let mut x = 0;
                while x < w {
                    acc = _mm256_add_epi32(acc, satd_quad(ra.add(x), sa, rb.add(x), sb));
                    x += 16;
                }
                y += 4;
            }
            // [A, A, B, B | C, C, D, D]: fold the lanes, then take one of each.
            let s = _mm_add_epi32(
                _mm256_castsi256_si128(acc),
                _mm256_extracti128_si256(acc, 1),
            );
            (_mm_cvtsi128_si32(s) as u32)
                .wrapping_add(_mm_cvtsi128_si32(_mm_srli_si128(s, 8)) as u32)
        }
    }

    fn satd(a: &[u8], a_stride: usize, b: &[u8], b_stride: usize, w: usize, h: usize) -> u32 {
        // Sixteen or more wide, or eight wide in whole 8x8 blocks; the rest
        // is a pair of tiles or fewer, one 128-bit vector's work.
        let ours = if w == 8 {
            h.is_multiple_of(8)
        } else {
            w.is_multiple_of(16) && h.is_multiple_of(4)
        };
        if !ours || h == 0 {
            return super::avx::satd(a, a_stride, b, b_stride, w, h);
        }
        assert!(
            a.len() >= (h - 1) * a_stride + w && b.len() >= (h - 1) * b_stride + w,
            "block out of range"
        );
        unsafe { satd_impl(a.as_ptr(), a_stride, b.as_ptr(), b_stride, w, h) }
    }
}

/// Install the best 8-bit distortion kernels `cpu` can run, one rung at a
/// time. SSE4.1 is not a rung here: nothing in these kernels has a better
/// SSE4.1 instruction, so an SSE4.1 CPU keeps SSSE3's and takes AVX's when
/// it has VEX.
pub fn install(d: &mut DistortionDsp<u8>, cpu: Cpu) {
    if cpu.sse2 {
        sse2::install_all(d);
    }
    if cpu.ssse3 {
        ssse3::install_satd(d);
    }
    if cpu.avx {
        avx::install_all(d);
    }
    if cpu.avx2 {
        avx2::install(d);
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

    /// Every rung the host can run, built cumulatively as `install` builds
    /// it in the field.
    fn rungs() -> Vec<(&'static str, DistortionDsp<u8>)> {
        let base = Cpu::SCALAR;
        [
            (
                "sse2",
                Cpu { sse2: true, ..base },
                std::is_x86_feature_detected!("sse2"),
            ),
            (
                "ssse3",
                Cpu {
                    sse2: true,
                    ssse3: true,
                    ..base
                },
                std::is_x86_feature_detected!("ssse3"),
            ),
            (
                "avx",
                Cpu {
                    sse2: true,
                    ssse3: true,
                    sse41: true,
                    avx: true,
                    ..base
                },
                std::is_x86_feature_detected!("avx"),
            ),
            (
                "avx2",
                Cpu {
                    sse2: true,
                    ssse3: true,
                    sse41: true,
                    avx: true,
                    avx2: true,
                    ..base
                },
                std::is_x86_feature_detected!("avx2"),
            ),
        ]
        .into_iter()
        .filter(|&(_, _, have)| have)
        .map(|(n, c, _)| {
            let mut d = DistortionDsp::<u8>::scalar();
            install(&mut d, c);
            (n, d)
        })
        .collect()
    }

    /// Block shapes both encoders ask for, plus a few they do not, so the
    /// remainders (a lone four-wide column, a width of twelve) are reached.
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

    /// Two planes: random bytes, with whole rows pinned to 0 or 255 now
    /// and then so the extremes (a difference of ±255 in every lane, a
    /// Hadamard coefficient of ±4080) are actually exercised.
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
    fn every_rung_matches_scalar() {
        let s = DistortionDsp::<u8>::scalar();
        let mut seed = 0x5add_u64;
        for (name, d) in rungs() {
            for round in 0..24 {
                let (a, b) = planes(&mut seed);
                for &(w, h) in &SIZES {
                    let sa = w + (lcg(&mut seed) as usize % 24);
                    let sb = w + (lcg(&mut seed) as usize % 24);
                    let oa = lcg(&mut seed) as usize % 64;
                    let ob = lcg(&mut seed) as usize % 64;
                    let (pa, pb) = (&a[oa..], &b[ob..]);
                    assert_eq!(
                        (d.sad)(pa, sa, pb, sb, w, h),
                        (s.sad)(pa, sa, pb, sb, w, h),
                        "{name} sad {w}x{h} round {round}"
                    );
                    assert_eq!(
                        (d.ssd)(pa, sa, pb, sb, w, h),
                        (s.ssd)(pa, sa, pb, sb, w, h),
                        "{name} ssd {w}x{h} round {round}"
                    );
                    assert_eq!(
                        (d.satd)(pa, sa, pb, sb, w, h),
                        (s.satd)(pa, sa, pb, sb, w, h),
                        "{name} satd {w}x{h} round {round}"
                    );
                }
            }
        }
    }

    /// The saturating extremes as a closed form, so the test does not rest
    /// on the scalar reference alone: a 64x64 block of 0 against 255.
    #[test]
    fn full_deflection_is_exact() {
        let a = vec![0u8; 64 * 64];
        let b = vec![255u8; 64 * 64];
        for (name, d) in rungs() {
            assert_eq!((d.sad)(&a, 64, &b, 64, 64, 64), 255 * 4096, "{name}");
            assert_eq!(
                (d.ssd)(&a, 64, &b, 64, 64, 64),
                255u64 * 255 * 4096,
                "{name}"
            );
            assert_eq!(
                (d.satd)(&a, 64, &b, 64, 64, 64),
                256 * ((16 * 255 + 1) >> 1),
                "{name}"
            );
        }
    }

    /// `DistortionDsp::new` reaches these through the sample-type dispatch,
    /// and does not hand the 8-bit kernels to a 16-bit table (that one has
    /// its own, `distortion_x86_u16`).
    #[test]
    fn new_installs_the_kernels_of_its_sample_type() {
        let cpu = Cpu::detect();
        let d8 = DistortionDsp::<u8>::new(cpu);
        if cpu.sse2 {
            assert!(
                d8.sad as usize != DistortionDsp::<u8>::scalar().sad as usize,
                "u8 sad still scalar"
            );
        }
        let d16 = DistortionDsp::<u16>::new(cpu);
        type Sad8 = crate::dsp::distortion::SadFn<u8>;
        let ours = [
            sse2::sad as Sad8 as usize,
            ssse3::sad as Sad8 as usize,
            avx::sad as Sad8 as usize,
        ];
        assert!(
            !ours.contains(&(d16.sad as usize)),
            "a u16 table took an 8-bit kernel"
        );
    }

    /// Cycles per call, scalar against each rung, over the shapes the
    /// encoders use. Not a correctness test: `cargo test --release
    /// distortion_x86 -- --ignored --nocapture` prints it.
    #[test]
    #[ignore]
    fn kernel_bench() {
        use std::time::Instant;
        let s = DistortionDsp::<u8>::scalar();
        let mut seed = 0xbe9c_u64;
        let (a, b) = planes(&mut seed);
        let shapes = [(4, 4), (8, 8), (16, 16), (32, 32), (64, 64)];
        let mut tables = vec![("scalar", s.clone()), ("scalar-again", s)];
        tables.extend(rungs());
        for &(w, h) in &shapes {
            let iters = 2_000_000 / (w * h) * 16;
            for (name, d) in &tables {
                let mut sink = 0u64;
                let t = Instant::now();
                for i in 0..iters {
                    let o = (i & 31) * 3;
                    sink = sink
                        .wrapping_add((d.sad)(&a[o..], 96, &b[o..], 96, w, h) as u64)
                        .wrapping_add((d.satd)(&a[o..], 96, &b[o..], 96, w, h) as u64)
                        .wrapping_add((d.ssd)(&a[o..], 96, &b[o..], 96, w, h));
                }
                let ns = t.elapsed().as_nanos() as f64 / iters as f64;
                println!(
                    "{w}x{h} {name:13} {ns:8.1} ns per (sad+satd+ssd) [{}]",
                    sink & 1
                );
            }
        }
    }
}
