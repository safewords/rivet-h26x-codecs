//! AArch64 NEON versions of the H.264 forward transforms and quantisers —
//! [`super::h264_enc_x86`]'s 128-bit kernels, on the instructions this
//! architecture has for them.
//!
//! The transforms are that file's lane-wise butterflies on `int32x4_t`,
//! with the 4x4 transposes as `trn1` / `trn2` at 32 and then 64 bits. The
//! quantisers compute the reference's `(|c| * mf + offset) >> qbits` in 64
//! bits: `umull` / `umull2` are the four 32 x 32 products, `ushl` by a
//! negative count the logical shift, `uzp1` the low halves back in lane
//! order. The level is truncated to i16 the way the reference's casts
//! truncate (`xtn`, which keeps the low half rather than saturating), and
//! a negative multiplier or offset, or a shift past 62, is the
//! reference's.
//!
//! NEON is baseline on AArch64. Checked bit-exact by the tests below, which
//! run under arm64 emulation (Docker) as well as on arm64 hardware.

#![cfg(target_arch = "aarch64")]

use std::arch::aarch64::*;

use super::Cpu;
use super::h264_enc::{H264EncDsp, quant4_scalar, quant8_scalar};

/// Install the NEON kernels.
pub fn install(d: &mut H264EncDsp, cpu: Cpu) {
    if cpu.neon {
        d.fdct4 = fdct4;
        d.fdct8 = fdct8;
        d.hadamard4 = hadamard4;
        d.quant4 = quant4;
        d.quant8 = quant8;
        d.satd_avg = Some(satd_avg);
        d.sad_x4 = Some(sad_x4);
        d.satd_avg_x4 = Some(satd_avg_x4);
        d.satd_x4 = Some(satd_x4);
        d.satd_bi_x4 = Some(satd_bi_x4);
        d.sub_fdct4 = Some(sub_fdct4);
        d.sub_fdct8 = Some(sub_fdct8);
        d.hpel_planes = Some(hpel_planes);
    }
}

/// [`super::h264_enc::HpelPlanesFn`]: sixteen horizontal or vertical half
/// samples a step in sixteen-bit lanes (an 8-bit six-tap sum is within
/// -2550..=10710), the vertical sums kept for the centre samples, which
/// take the six-tap over them in thirty-two-bit lanes; the rounding shift
/// and the clip to 0..=255 one saturating narrow each (`sqrshrun`). A row
/// whose length is not a multiple of the step ends with a step that
/// overlaps the one before (the same values written again). The border
/// columns and rows the six-tap cannot reach copy the integer samples.
fn hpel_planes(src: &[u8], stride: usize, h: &mut [u8], v: &mut [u8], c: &mut [u8]) {
    let len = src.len();
    let rows = len / stride;
    if stride < 24 || rows < 6 || len % stride != 0 {
        return super::h264_enc::hpel_planes_scalar(src, stride, h, v, c);
    }
    assert!(h.len() == len && v.len() == len && c.len() == len);
    let mut inter = vec![0i16; stride];
    // SAFETY: every access below is inside a row of `stride` samples of
    // planes `len` long: horizontal steps read `x - 2 .. x + 19` for `x`
    // in `2 ..= stride - 19`, vertical ones rows `y - 2 ..= y + 3` for `y`
    // in `2 .. rows - 3`.
    unsafe {
        for y in 0..rows {
            let row = src.as_ptr().add(y * stride);
            let o = h.as_mut_ptr().add(y * stride);
            hpel_h_row(row, o, stride);
            // The border: two columns before, three after.
            for x in [0, 1, stride - 3, stride - 2, stride - 1] {
                *o.add(x) = *row.add(x);
            }
        }
        for y in 0..rows {
            let (vo, co) = (
                v.as_mut_ptr().add(y * stride),
                c.as_mut_ptr().add(y * stride),
            );
            let row = src.as_ptr().add(y * stride);
            if y < 2 || y + 3 >= rows {
                std::ptr::copy_nonoverlapping(row, vo, stride);
                std::ptr::copy_nonoverlapping(row, co, stride);
                continue;
            }
            hpel_v_row(
                src.as_ptr().add((y - 2) * stride),
                stride,
                vo,
                inter.as_mut_ptr(),
            );
            hpel_c_row(inter.as_ptr(), co, stride);
            for x in [0, 1, stride - 3, stride - 2, stride - 1] {
                *co.add(x) = *row.add(x);
            }
        }
    }
}

/// `a + f - 5 (b + e) + 20 (c + d)` of eight-bit lanes, in sixteen bits.
#[inline(always)]
unsafe fn tap6_u8(
    a: uint8x8_t,
    b: uint8x8_t,
    c: uint8x8_t,
    d: uint8x8_t,
    e: uint8x8_t,
    f: uint8x8_t,
) -> int16x8_t {
    unsafe {
        let t = vaddl_u8(a, f);
        let t = vmlaq_n_u16(t, vaddl_u8(c, d), 20);
        let t = vmlsq_n_u16(t, vaddl_u8(b, e), 5);
        vreinterpretq_s16_u16(t)
    }
}

/// One row's horizontal half samples at `2 .. stride - 3`.
#[target_feature(enable = "neon")]
unsafe fn hpel_h_row(row: *const u8, out: *mut u8, stride: usize) {
    unsafe {
        let end = stride - 3;
        let mut x = 2;
        loop {
            let x0 = x.min(end - 16);
            let l = |k: usize| vld1q_u8(row.add(x0 - 2 + k));
            let (a, b, c, d, e, f) = (l(0), l(1), l(2), l(3), l(4), l(5));
            let lo = tap6_u8(
                vget_low_u8(a),
                vget_low_u8(b),
                vget_low_u8(c),
                vget_low_u8(d),
                vget_low_u8(e),
                vget_low_u8(f),
            );
            let hi = tap6_u8(
                vget_high_u8(a),
                vget_high_u8(b),
                vget_high_u8(c),
                vget_high_u8(d),
                vget_high_u8(e),
                vget_high_u8(f),
            );
            vst1q_u8(
                out.add(x0),
                vcombine_u8(vqrshrun_n_s16::<5>(lo), vqrshrun_n_s16::<5>(hi)),
            );
            if x0 + 16 >= end {
                break;
            }
            x += 16;
        }
    }
}

/// One row's vertical half samples (every column) from the six rows from
/// `top`, and their sums into `inter`.
#[target_feature(enable = "neon")]
unsafe fn hpel_v_row(top: *const u8, stride: usize, out: *mut u8, inter: *mut i16) {
    unsafe {
        let mut x = 0;
        loop {
            let x0 = x.min(stride - 16);
            let l = |k: usize| vld1q_u8(top.add(k * stride + x0));
            let (a, b, c, d, e, f) = (l(0), l(1), l(2), l(3), l(4), l(5));
            let lo = tap6_u8(
                vget_low_u8(a),
                vget_low_u8(b),
                vget_low_u8(c),
                vget_low_u8(d),
                vget_low_u8(e),
                vget_low_u8(f),
            );
            let hi = tap6_u8(
                vget_high_u8(a),
                vget_high_u8(b),
                vget_high_u8(c),
                vget_high_u8(d),
                vget_high_u8(e),
                vget_high_u8(f),
            );
            vst1q_s16(inter.add(x0), lo);
            vst1q_s16(inter.add(x0 + 8), hi);
            vst1q_u8(
                out.add(x0),
                vcombine_u8(vqrshrun_n_s16::<5>(lo), vqrshrun_n_s16::<5>(hi)),
            );
            if x0 + 16 >= stride {
                break;
            }
            x += 16;
        }
    }
}

/// `a + f - 5 (b + e) + 20 (c + d)` of sixteen-bit lanes, in thirty-two.
#[inline(always)]
unsafe fn tap6_s16(
    a: int16x4_t,
    b: int16x4_t,
    c: int16x4_t,
    d: int16x4_t,
    e: int16x4_t,
    f: int16x4_t,
) -> int32x4_t {
    unsafe {
        let t = vaddl_s16(a, f);
        let t = vmlaq_n_s32(t, vaddl_s16(c, d), 20);
        vmlsq_n_s32(t, vaddl_s16(b, e), 5)
    }
}

/// One row's centre half samples at `2 .. stride - 3` from the vertical
/// sums.
#[target_feature(enable = "neon")]
unsafe fn hpel_c_row(inter: *const i16, out: *mut u8, stride: usize) {
    unsafe {
        let end = stride - 3;
        let mut x = 2;
        loop {
            let x0 = x.min(end - 8);
            let l = |k: usize| vld1q_s16(inter.add(x0 - 2 + k));
            let (a, b, c, d, e, f) = (l(0), l(1), l(2), l(3), l(4), l(5));
            let lo = tap6_s16(
                vget_low_s16(a),
                vget_low_s16(b),
                vget_low_s16(c),
                vget_low_s16(d),
                vget_low_s16(e),
                vget_low_s16(f),
            );
            let hi = tap6_s16(
                vget_high_s16(a),
                vget_high_s16(b),
                vget_high_s16(c),
                vget_high_s16(d),
                vget_high_s16(e),
                vget_high_s16(f),
            );
            let n = vcombine_u16(vqrshrun_n_s32::<10>(lo), vqrshrun_n_s32::<10>(hi));
            vst1_u8(out.add(x0), vqmovn_u16(n));
            if x0 + 8 >= end {
                break;
            }
            x += 8;
        }
    }
}

/// [`super::h264_enc::SadX4Fn`]: each source row loaded once and its
/// absolute differences against the four candidates' rows accumulated in
/// sixteen-bit lanes (a 16-wide block's lane sums at most 2 * 255 a row,
/// so 64 rows fit), widened once at the end.
fn sad_x4(src: &[u8], ss: usize, refs: [&[u8]; 4], rs: usize, w: usize, h: usize) -> [u32; 4] {
    if !(w == 8 || w == 16) || h == 0 || h > 64 {
        return super::h264_enc::sad_x4_scalar(src, ss, refs, rs, w, h);
    }
    let need = (h - 1) * rs + w;
    assert!(
        src.len() >= (h - 1) * ss + w && refs.iter().all(|r| r.len() >= need),
        "block out of range"
    );
    let p = refs.map(|r| r.as_ptr());
    unsafe {
        if w == 16 {
            sad_x4_16(src.as_ptr(), ss, p, rs, h)
        } else {
            sad_x4_8(src.as_ptr(), ss, p, rs, h)
        }
    }
}

#[target_feature(enable = "neon")]
unsafe fn sad_x4_16(s: *const u8, ss: usize, r: [*const u8; 4], rs: usize, h: usize) -> [u32; 4] {
    unsafe {
        let mut acc = [vdupq_n_u16(0); 4];
        for y in 0..h {
            let v = vld1q_u8(s.add(y * ss));
            for k in 0..4 {
                acc[k] = vpadalq_u8(acc[k], vabdq_u8(v, vld1q_u8(r[k].add(y * rs))));
            }
        }
        acc.map(|a| vaddlvq_u16(a))
    }
}

#[target_feature(enable = "neon")]
unsafe fn sad_x4_8(s: *const u8, ss: usize, r: [*const u8; 4], rs: usize, h: usize) -> [u32; 4] {
    unsafe {
        let mut acc = [vdupq_n_u16(0); 4];
        for y in 0..h {
            let v = vld1_u8(s.add(y * ss));
            for k in 0..4 {
                acc[k] = vabal_u8(acc[k], v, vld1_u8(r[k].add(y * rs)));
            }
        }
        acc.map(|a| vaddlvq_u16(a))
    }
}

/// [`super::h264_enc::SatdAvgFn`]: each prediction row is `urhadd` of the
/// two references' rows, differenced against the source and fed to the
/// distortion kernel's tile pairs — the SATD of a quarter-sample candidate
/// with no buffer in between.
#[allow(clippy::too_many_arguments)]
fn satd_avg(
    src: &[u8],
    ss: usize,
    a: &[u8],
    sa: usize,
    b: &[u8],
    sb: usize,
    w: usize,
    h: usize,
) -> u32 {
    if !w.is_multiple_of(4) || !h.is_multiple_of(4) || h == 0 || w > 16 {
        return super::h264_enc::satd_avg_scalar(src, ss, a, sa, b, sb, w, h);
    }
    assert!(
        src.len() >= (h - 1) * ss + w && a.len() >= (h - 1) * sa + w && b.len() >= (h - 1) * sb + w,
        "block out of range"
    );
    unsafe { satd_avg_impl(src.as_ptr(), ss, a.as_ptr(), sa, b.as_ptr(), sb, w, h) }
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn satd_avg_impl(
    s: *const u8,
    ss: usize,
    a: *const u8,
    sa: usize,
    b: *const u8,
    sb: usize,
    w: usize,
    h: usize,
) -> u32 {
    use super::distortion_neon::{acc_pair, diff8, load4, load4x2, satd_pair};
    unsafe {
        let mut acc = vdupq_n_s32(0);
        if w == 4 {
            let mut y = 0;
            while y + 8 <= h {
                let row = |r: usize| {
                    let p = vrhadd_u8(
                        load4x2(a.add((y + r) * sa), a.add((y + r + 4) * sa)),
                        load4x2(b.add((y + r) * sb), b.add((y + r + 4) * sb)),
                    );
                    diff8(load4x2(s.add((y + r) * ss), s.add((y + r + 4) * ss)), p)
                };
                acc = acc_pair(acc, satd_pair(row(0), row(1), row(2), row(3)));
                y += 8;
            }
            if y < h {
                let row = |r: usize| {
                    let p = vrhadd_u8(load4(a.add((y + r) * sa)), load4(b.add((y + r) * sb)));
                    diff8(load4(s.add((y + r) * ss)), p)
                };
                acc = acc_pair(acc, satd_pair(row(0), row(1), row(2), row(3)));
            }
        } else {
            let mut y = 0;
            while y < h {
                let (rs, ra, rb) = (s.add(y * ss), a.add(y * sa), b.add(y * sb));
                let mut x = 0;
                // Sixteen columns a row: one load each, two tile pairs.
                while x + 16 <= w {
                    let row = |r: usize| {
                        let p =
                            vrhaddq_u8(vld1q_u8(ra.add(r * sa + x)), vld1q_u8(rb.add(r * sb + x)));
                        let v = vld1q_u8(rs.add(r * ss + x));
                        (
                            vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(v), vget_low_u8(p))),
                            vreinterpretq_s16_u16(vsubl_high_u8(v, p)),
                        )
                    };
                    let (r0, r1, r2, r3) = (row(0), row(1), row(2), row(3));
                    acc = acc_pair(acc, satd_pair(r0.0, r1.0, r2.0, r3.0));
                    acc = acc_pair(acc, satd_pair(r0.1, r1.1, r2.1, r3.1));
                    x += 16;
                }
                while x + 8 <= w {
                    let row = |r: usize| {
                        let p = vrhadd_u8(vld1_u8(ra.add(r * sa + x)), vld1_u8(rb.add(r * sb + x)));
                        diff8(vld1_u8(rs.add(r * ss + x)), p)
                    };
                    acc = acc_pair(acc, satd_pair(row(0), row(1), row(2), row(3)));
                    x += 8;
                }
                if x < w {
                    let row = |r: usize| {
                        let p = vrhadd_u8(load4(ra.add(r * sa + x)), load4(rb.add(r * sb + x)));
                        diff8(load4(rs.add(r * ss + x)), p)
                    };
                    acc = acc_pair(acc, satd_pair(row(0), row(1), row(2), row(3)));
                }
                y += 4;
            }
        }
        vaddvq_s32(acc) as u32
    }
}

/// [`super::h264_enc::SatdAvgX4Fn`]: [`satd_avg_impl`]'s rows for four
/// candidates, each source row loaded once per stripe of four and
/// differenced against the four rounded averages in turn.
fn satd_avg_x4(
    src: &[u8],
    ss: usize,
    pairs: [(&[u8], &[u8]); 4],
    rs: usize,
    w: usize,
    h: usize,
) -> [u32; 4] {
    if !(w == 8 || w == 16) || !h.is_multiple_of(4) || h == 0 {
        return super::h264_enc::satd_avg_x4_scalar(src, ss, pairs, rs, w, h);
    }
    let need = (h - 1) * rs + w;
    assert!(
        src.len() >= (h - 1) * ss + w
            && pairs
                .iter()
                .all(|(a, b)| a.len() >= need && b.len() >= need),
        "block out of range"
    );
    let p = pairs.map(|(a, b)| (a.as_ptr(), b.as_ptr()));
    // SAFETY: NEON is baseline on AArch64; every row read is inside the
    // blocks checked above.
    unsafe {
        if w == 16 {
            satd_avg_x4_16::<true>(src.as_ptr(), ss, p, rs, h)
        } else {
            satd_avg_x4_8::<true>(src.as_ptr(), ss, p, rs, h)
        }
    }
}

/// [`super::h264_enc::SatdX4Fn`]: [`satd_avg_x4`]'s kernels with the
/// candidates' rows read as they are, no second view loaded or averaged.
fn satd_x4(src: &[u8], ss: usize, refs: [&[u8]; 4], rs: usize, w: usize, h: usize) -> [u32; 4] {
    if !(w == 8 || w == 16) || !h.is_multiple_of(4) || h == 0 {
        return super::h264_enc::satd_x4_scalar(src, ss, refs, rs, w, h);
    }
    let need = (h - 1) * rs + w;
    assert!(
        src.len() >= (h - 1) * ss + w && refs.iter().all(|r| r.len() >= need),
        "block out of range"
    );
    let p = refs.map(|r| (r.as_ptr(), r.as_ptr()));
    // SAFETY: NEON is baseline on AArch64; every row read is inside the
    // blocks checked above (the second view of each pair is never read).
    unsafe {
        if w == 16 {
            satd_avg_x4_16::<false>(src.as_ptr(), ss, p, rs, h)
        } else {
            satd_avg_x4_8::<false>(src.as_ptr(), ss, p, rs, h)
        }
    }
}

/// `AVG`: each candidate's row the rounded average of its pair's rows;
/// otherwise the first view's row alone.
#[target_feature(enable = "neon")]
unsafe fn satd_avg_x4_16<const AVG: bool>(
    s: *const u8,
    ss: usize,
    p: [(*const u8, *const u8); 4],
    rs: usize,
    h: usize,
) -> [u32; 4] {
    use super::distortion_neon::{acc_pair, satd_pair};
    unsafe {
        let mut acc = [vdupq_n_s32(0); 4];
        let mut y = 0;
        while y < h {
            let v = [0, 1, 2, 3].map(|r| vld1q_u8(s.add((y + r) * ss)));
            for k in 0..4 {
                let (a, b) = p[k];
                let row = |r: usize| {
                    let o = (y + r) * rs;
                    let q = if AVG {
                        vrhaddq_u8(vld1q_u8(a.add(o)), vld1q_u8(b.add(o)))
                    } else {
                        vld1q_u8(a.add(o))
                    };
                    (
                        vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(v[r]), vget_low_u8(q))),
                        vreinterpretq_s16_u16(vsubl_high_u8(v[r], q)),
                    )
                };
                let (r0, r1, r2, r3) = (row(0), row(1), row(2), row(3));
                acc[k] = acc_pair(acc[k], satd_pair(r0.0, r1.0, r2.0, r3.0));
                acc[k] = acc_pair(acc[k], satd_pair(r0.1, r1.1, r2.1, r3.1));
            }
            y += 4;
        }
        acc.map(|a| vaddvq_s32(a) as u32)
    }
}

/// [`satd_avg_x4_16`] for eight columns.
#[target_feature(enable = "neon")]
unsafe fn satd_avg_x4_8<const AVG: bool>(
    s: *const u8,
    ss: usize,
    p: [(*const u8, *const u8); 4],
    rs: usize,
    h: usize,
) -> [u32; 4] {
    use super::distortion_neon::{acc_pair, diff8, satd_pair};
    unsafe {
        let mut acc = [vdupq_n_s32(0); 4];
        let mut y = 0;
        while y < h {
            let v = [0, 1, 2, 3].map(|r| vld1_u8(s.add((y + r) * ss)));
            for k in 0..4 {
                let (a, b) = p[k];
                let row = |r: usize| {
                    let o = (y + r) * rs;
                    let q = if AVG {
                        vrhadd_u8(vld1_u8(a.add(o)), vld1_u8(b.add(o)))
                    } else {
                        vld1_u8(a.add(o))
                    };
                    diff8(v[r], q)
                };
                acc[k] = acc_pair(acc[k], satd_pair(row(0), row(1), row(2), row(3)));
            }
            y += 4;
        }
        acc.map(|a| vaddvq_s32(a) as u32)
    }
}

/// [`super::h264_enc::SatdBiX4Fn`]: [`satd_avg_x4`] with each candidate's
/// average averaged again with the held prediction's row.
#[allow(clippy::too_many_arguments)]
fn satd_bi_x4(
    src: &[u8],
    ss: usize,
    pairs: [(&[u8], &[u8]); 4],
    rs: usize,
    held: &[u8],
    hs: usize,
    w: usize,
    h: usize,
) -> [u32; 4] {
    if !(w == 8 || w == 16) || !h.is_multiple_of(4) || h == 0 {
        return super::h264_enc::satd_bi_x4_scalar(src, ss, pairs, rs, held, hs, w, h);
    }
    let need = (h - 1) * rs + w;
    assert!(
        src.len() >= (h - 1) * ss + w
            && held.len() >= (h - 1) * hs + w
            && pairs
                .iter()
                .all(|(a, b)| a.len() >= need && b.len() >= need),
        "block out of range"
    );
    let p = pairs.map(|(a, b)| (a.as_ptr(), b.as_ptr()));
    // SAFETY: NEON is baseline on AArch64; every row read is inside the
    // blocks checked above.
    unsafe {
        if w == 16 {
            satd_bi_x4_16(src.as_ptr(), ss, p, rs, held.as_ptr(), hs, h)
        } else {
            satd_bi_x4_8(src.as_ptr(), ss, p, rs, held.as_ptr(), hs, h)
        }
    }
}

#[target_feature(enable = "neon")]
unsafe fn satd_bi_x4_16(
    s: *const u8,
    ss: usize,
    p: [(*const u8, *const u8); 4],
    rs: usize,
    hp: *const u8,
    hs: usize,
    h: usize,
) -> [u32; 4] {
    use super::distortion_neon::{acc_pair, satd_pair};
    unsafe {
        let mut acc = [vdupq_n_s32(0); 4];
        let mut y = 0;
        while y < h {
            let v = [0, 1, 2, 3].map(|r| vld1q_u8(s.add((y + r) * ss)));
            let hv = [0, 1, 2, 3].map(|r| vld1q_u8(hp.add((y + r) * hs)));
            for k in 0..4 {
                let (a, b) = p[k];
                let row = |r: usize| {
                    let o = (y + r) * rs;
                    let q = vrhaddq_u8(vrhaddq_u8(vld1q_u8(a.add(o)), vld1q_u8(b.add(o))), hv[r]);
                    (
                        vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(v[r]), vget_low_u8(q))),
                        vreinterpretq_s16_u16(vsubl_high_u8(v[r], q)),
                    )
                };
                let (r0, r1, r2, r3) = (row(0), row(1), row(2), row(3));
                acc[k] = acc_pair(acc[k], satd_pair(r0.0, r1.0, r2.0, r3.0));
                acc[k] = acc_pair(acc[k], satd_pair(r0.1, r1.1, r2.1, r3.1));
            }
            y += 4;
        }
        acc.map(|a| vaddvq_s32(a) as u32)
    }
}

#[target_feature(enable = "neon")]
unsafe fn satd_bi_x4_8(
    s: *const u8,
    ss: usize,
    p: [(*const u8, *const u8); 4],
    rs: usize,
    hp: *const u8,
    hs: usize,
    h: usize,
) -> [u32; 4] {
    use super::distortion_neon::{acc_pair, diff8, satd_pair};
    unsafe {
        let mut acc = [vdupq_n_s32(0); 4];
        let mut y = 0;
        while y < h {
            let v = [0, 1, 2, 3].map(|r| vld1_u8(s.add((y + r) * ss)));
            let hv = [0, 1, 2, 3].map(|r| vld1_u8(hp.add((y + r) * hs)));
            for k in 0..4 {
                let (a, b) = p[k];
                let row = |r: usize| {
                    let o = (y + r) * rs;
                    diff8(
                        v[r],
                        vrhadd_u8(vrhadd_u8(vld1_u8(a.add(o)), vld1_u8(b.add(o))), hv[r]),
                    )
                };
                acc[k] = acc_pair(acc[k], satd_pair(row(0), row(1), row(2), row(3)));
            }
            y += 4;
        }
        acc.map(|a| vaddvq_s32(a) as u32)
    }
}

/// Transpose a 4x4 block of i32, one row a vector.
#[inline(always)]
unsafe fn transpose4(r: [int32x4_t; 4]) -> [int32x4_t; 4] {
    unsafe {
        let t0 = vtrn1q_s32(r[0], r[1]);
        let t1 = vtrn2q_s32(r[0], r[1]);
        let t2 = vtrn1q_s32(r[2], r[3]);
        let t3 = vtrn2q_s32(r[2], r[3]);
        let w = |v: int32x4_t| vreinterpretq_s64_s32(v);
        let n = |v: int64x2_t| vreinterpretq_s32_s64(v);
        [
            n(vtrn1q_s64(w(t0), w(t2))),
            n(vtrn1q_s64(w(t1), w(t3))),
            n(vtrn2q_s64(w(t0), w(t2))),
            n(vtrn2q_s64(w(t1), w(t3))),
        ]
    }
}

/// `fdct4_1d`, lane-wise.
#[inline(always)]
unsafe fn fdct4_lanes(x: [int32x4_t; 4]) -> [int32x4_t; 4] {
    unsafe {
        let s0 = vaddq_s32(x[0], x[3]);
        let s1 = vaddq_s32(x[1], x[2]);
        let s2 = vsubq_s32(x[1], x[2]);
        let s3 = vsubq_s32(x[0], x[3]);
        [
            vaddq_s32(s0, s1),
            vaddq_s32(vaddq_s32(s3, s3), s2),
            vsubq_s32(s0, s1),
            vsubq_s32(s3, vaddq_s32(s2, s2)),
        ]
    }
}

/// `had4_1d`, lane-wise.
#[inline(always)]
unsafe fn had4_lanes(x: [int32x4_t; 4]) -> [int32x4_t; 4] {
    unsafe {
        let s0 = vaddq_s32(x[0], x[3]);
        let s1 = vaddq_s32(x[1], x[2]);
        let s2 = vsubq_s32(x[1], x[2]);
        let s3 = vsubq_s32(x[0], x[3]);
        [
            vaddq_s32(s0, s1),
            vaddq_s32(s3, s2),
            vsubq_s32(s0, s1),
            vsubq_s32(s3, s2),
        ]
    }
}

fn fdct4(residual: &[i16; 16], coeffs: &mut [i32; 16]) {
    unsafe {
        let p = residual.as_ptr();
        let row = |i: usize| vmovl_s16(vld1_s16(p.add(4 * i)));
        // Columns as vectors for the row pass, rows for the column pass.
        let t = fdct4_lanes(transpose4([row(0), row(1), row(2), row(3)]));
        let o = fdct4_lanes(transpose4(t));
        let q = coeffs.as_mut_ptr();
        for (i, v) in o.iter().enumerate() {
            vst1q_s32(q.add(4 * i), *v);
        }
    }
}

/// [`super::h264_enc::SubFdct4Fn`]: each row's difference widened
/// straight from the bytes (`usubl`) into [`fdct4`]'s lanes.
fn sub_fdct4(src: &[u8], ss: usize, pred: &[u8], ps: usize, coeffs: &mut [i32; 16]) {
    assert!(
        src.len() >= 3 * ss + 4 && pred.len() >= 3 * ps + 4,
        "block out of range"
    );
    unsafe {
        use super::distortion_neon::load4;
        let (s, p) = (src.as_ptr(), pred.as_ptr());
        let row = |i: usize| {
            let d = vreinterpretq_s16_u16(vsubl_u8(load4(s.add(i * ss)), load4(p.add(i * ps))));
            vmovl_s16(vget_low_s16(d))
        };
        let t = fdct4_lanes(transpose4([row(0), row(1), row(2), row(3)]));
        let o = fdct4_lanes(transpose4(t));
        let q = coeffs.as_mut_ptr();
        for (i, v) in o.iter().enumerate() {
            vst1q_s32(q.add(4 * i), *v);
        }
    }
}

/// [`super::h264_enc::SubFdct8Fn`]: the difference rows straight into
/// [`fdct8`]'s residual, one `usubl` each, then that transform.
fn sub_fdct8(src: &[u8], ss: usize, pred: &[u8], ps: usize, coeffs: &mut [i32; 64]) {
    assert!(
        src.len() >= 7 * ss + 8 && pred.len() >= 7 * ps + 8,
        "block out of range"
    );
    let mut r = [0i16; 64];
    unsafe {
        let (s, p) = (src.as_ptr(), pred.as_ptr());
        for i in 0..8 {
            let d = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(s.add(i * ss)), vld1_u8(p.add(i * ps))));
            vst1q_s16(r.as_mut_ptr().add(8 * i), d);
        }
    }
    fdct8(&r, coeffs);
}

fn hadamard4(dc: &mut [i32; 16]) {
    unsafe {
        let p = dc.as_mut_ptr();
        let row = |i: usize| vld1q_s32(p.add(4 * i));
        let t = had4_lanes(transpose4([row(0), row(1), row(2), row(3)]));
        let o = had4_lanes(transpose4(t));
        for (i, v) in o.iter().enumerate() {
            vst1q_s32(p.add(4 * i), *v);
        }
    }
}

/// `fdct8_1d`, lane-wise.
#[inline(always)]
unsafe fn fdct8_lanes(x: &[int32x4_t; 8]) -> [int32x4_t; 8] {
    unsafe {
        let a0 = vaddq_s32(x[0], x[7]);
        let a1 = vaddq_s32(x[1], x[6]);
        let a2 = vaddq_s32(x[2], x[5]);
        let a3 = vaddq_s32(x[3], x[4]);
        let a4 = vsubq_s32(x[0], x[7]);
        let a5 = vsubq_s32(x[1], x[6]);
        let a6 = vsubq_s32(x[2], x[5]);
        let a7 = vsubq_s32(x[3], x[4]);
        let b0 = vaddq_s32(a0, a3);
        let b1 = vaddq_s32(a1, a2);
        let b2 = vsubq_s32(a0, a3);
        let b3 = vsubq_s32(a1, a2);
        let h = |v: int32x4_t| vaddq_s32(vshrq_n_s32::<1>(v), v);
        let b4 = vaddq_s32(vaddq_s32(a5, a6), h(a4));
        let b5 = vsubq_s32(vsubq_s32(a4, a7), h(a6));
        let b6 = vsubq_s32(vaddq_s32(a4, a7), h(a5));
        let b7 = vaddq_s32(vsubq_s32(a5, a6), h(a7));
        [
            vaddq_s32(b0, b1),
            vaddq_s32(b4, vshrq_n_s32::<2>(b7)),
            vaddq_s32(b2, vshrq_n_s32::<1>(b3)),
            vaddq_s32(b5, vshrq_n_s32::<2>(b6)),
            vsubq_s32(b0, b1),
            vsubq_s32(b6, vshrq_n_s32::<2>(b5)),
            vsubq_s32(vshrq_n_s32::<1>(b2), b3),
            vsubq_s32(vshrq_n_s32::<2>(b4), b7),
        ]
    }
}

/// The 8x8 transpose of `q[half][k]` (row `4 * half + lane`, column `k`)
/// into the same layout with rows and columns swapped.
#[inline(always)]
unsafe fn transpose8x8(q: &[[int32x4_t; 8]; 2]) -> [[int32x4_t; 8]; 2] {
    unsafe {
        let mut out = [[vdupq_n_s32(0); 8]; 2];
        for rh in 0..2 {
            for cb in 0..2 {
                let t = transpose4([
                    q[rh][4 * cb],
                    q[rh][4 * cb + 1],
                    q[rh][4 * cb + 2],
                    q[rh][4 * cb + 3],
                ]);
                out[cb][4 * rh..4 * rh + 4].copy_from_slice(&t);
            }
        }
        out
    }
}

fn fdct8(residual: &[i16; 64], coeffs: &mut [i32; 64]) {
    unsafe {
        let p = residual.as_ptr();
        // `c[half][k]`: column `k` of rows `4 * half..`, sign-extended.
        let mut c = [[vdupq_n_s32(0); 8]; 2];
        for (half, ch) in c.iter_mut().enumerate() {
            let rows: [int16x8_t; 4] =
                std::array::from_fn(|i| vld1q_s16(p.add(8 * (4 * half + i))));
            let lo = transpose4(std::array::from_fn(|i| vmovl_s16(vget_low_s16(rows[i]))));
            let hi = transpose4(std::array::from_fn(|i| vmovl_high_s16(rows[i])));
            ch[..4].copy_from_slice(&lo);
            ch[4..].copy_from_slice(&hi);
        }
        let t = [fdct8_lanes(&c[0]), fdct8_lanes(&c[1])];
        let u = transpose8x8(&t);
        let o = [fdct8_lanes(&u[0]), fdct8_lanes(&u[1])];
        let q = coeffs.as_mut_ptr();
        for (i, (lo, hi)) in o[0].iter().zip(&o[1]).enumerate() {
            vst1q_s32(q.add(8 * i), *lo);
            vst1q_s32(q.add(8 * i + 4), *hi);
        }
    }
}

/// Quantise four coefficients: the levels, still i32.
#[inline(always)]
unsafe fn quant_lanes(c: int32x4_t, mf: uint32x4_t, off: uint64x2_t, sh: int64x2_t) -> int32x4_t {
    unsafe {
        // `abs` of i32::MIN wraps to 0x8000_0000: 2^31 as u32, `unsigned_abs`.
        let a = vreinterpretq_u32_s32(vabsq_s32(c));
        let lo = vshlq_u64(
            vaddq_u64(vmull_u32(vget_low_u32(a), vget_low_u32(mf)), off),
            sh,
        );
        let hi = vshlq_u64(vaddq_u64(vmull_high_u32(a, mf), off), sh);
        // The low 32 bits of each 64-bit result, in lane order.
        let m = vreinterpretq_s32_u32(vuzp1q_u32(
            vreinterpretq_u32_u64(lo),
            vreinterpretq_u32_u64(hi),
        ));
        let s = vshrq_n_s32::<31>(c);
        vsubq_s32(veorq_s32(m, s), s)
    }
}

/// `None` when a multiplier is negative (not a u32 one): the caller redoes
/// the block in the reference.
unsafe fn quant_impl(
    coeffs: *const i32,
    levels: *mut i16,
    mf: *const i32,
    n: usize,
    qbits: u32,
    offset: i32,
) -> Option<u32> {
    unsafe {
        let off = vdupq_n_u64(offset as u64);
        let sh = vdupq_n_s64(-(qbits as i64));
        let mut signs = vdupq_n_s32(0);
        let mut zeros = vdupq_n_u32(0);
        let mut i = 0;
        while i < n {
            let (m0, m1) = (vld1q_s32(mf.add(i)), vld1q_s32(mf.add(i + 4)));
            signs = vorrq_s32(signs, vorrq_s32(m0, m1));
            let v0 = quant_lanes(vld1q_s32(coeffs.add(i)), vreinterpretq_u32_s32(m0), off, sh);
            let v1 = quant_lanes(
                vld1q_s32(coeffs.add(i + 4)),
                vreinterpretq_u32_s32(m1),
                off,
                sh,
            );
            // `xtn` keeps each lane's low half: the reference's `as i16`.
            vst1q_s16(levels.add(i), vcombine_s16(vmovn_s32(v0), vmovn_s32(v1)));
            // One per zero level: subtract the all-ones masks.
            zeros = vsubq_u32(zeros, vceqzq_s32(v0));
            zeros = vsubq_u32(zeros, vceqzq_s32(v1));
            i += 8;
        }
        if vmaxvq_u32(vreinterpretq_u32_s32(vshrq_n_s32::<31>(signs))) != 0 {
            None
        } else {
            Some(n as u32 - vaddvq_u32(zeros))
        }
    }
}

fn quant4(
    coeffs: &[i32; 16],
    levels: &mut [i16; 16],
    mf: &[i32; 16],
    qbits: u32,
    offset: i32,
) -> u32 {
    if offset < 0 || qbits > 62 {
        return quant4_scalar(coeffs, levels, mf, qbits, offset);
    }
    match unsafe {
        quant_impl(
            coeffs.as_ptr(),
            levels.as_mut_ptr(),
            mf.as_ptr(),
            16,
            qbits,
            offset,
        )
    } {
        Some(nz) => nz,
        None => quant4_scalar(coeffs, levels, mf, qbits, offset),
    }
}

fn quant8(
    coeffs: &[i32; 64],
    levels: &mut [i16; 64],
    mf: &[i32; 64],
    qbits: u32,
    offset: i32,
) -> u32 {
    if offset < 0 || qbits > 62 {
        return quant8_scalar(coeffs, levels, mf, qbits, offset);
    }
    match unsafe {
        quant_impl(
            coeffs.as_ptr(),
            levels.as_mut_ptr(),
            mf.as_ptr(),
            64,
            qbits,
            offset,
        )
    } {
        Some(nz) => nz,
        None => quant8_scalar(coeffs, levels, mf, qbits, offset),
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn hpel_planes_match_scalar() {
        let mut seed = 7u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            (seed >> 16) as u8
        };
        for &(stride, rows) in &[(24usize, 6usize), (37, 9), (64, 20), (101, 13), (1328, 8)] {
            for extreme in 0..3 {
                let src: Vec<u8> = (0..stride * rows)
                    .map(|_| match extreme {
                        0 => rnd(),
                        // Alternating 0 / 255: the six-tap's extremes.
                        1 => (rnd() & 1) * 255,
                        _ => {
                            if rnd() & 1 == 0 {
                                0
                            } else {
                                255
                            }
                        }
                    })
                    .collect();
                let n = src.len();
                let (mut h0, mut v0, mut c0) = (vec![1u8; n], vec![2u8; n], vec![3u8; n]);
                let (mut h1, mut v1, mut c1) = (vec![4u8; n], vec![5u8; n], vec![6u8; n]);
                super::super::h264_enc::hpel_planes_scalar(&src, stride, &mut h0, &mut v0, &mut c0);
                hpel_planes(&src, stride, &mut h1, &mut v1, &mut c1);
                assert_eq!(h0, h1, "h {stride}x{rows}");
                assert_eq!(v0, v1, "v {stride}x{rows}");
                assert_eq!(c0, c1, "c {stride}x{rows}");
            }
        }
    }
    use super::*;
    use crate::dsp::h264_enc::{Quant, qbits4, qbits8, quant_offset};
    use crate::h264::sps::ScalingLists;

    fn lcg(seed: &mut u64) -> u32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*seed >> 33) as u32
    }

    /// The NEON table. NEON is baseline on AArch64, so this cannot be
    /// vacuous on the architecture it compiles for.
    fn neon() -> H264EncDsp {
        let mut d = H264EncDsp::SCALAR;
        install(
            &mut d,
            Cpu {
                neon: true,
                ..Cpu::SCALAR
            },
        );
        assert!(
            d.quant4 as usize != H264EncDsp::SCALAR.quant4 as usize,
            "NEON installed nothing"
        );
        d
    }

    /// The four-way SAD against its definition: the search's shapes, the
    /// tallest a kernel accumulates, and the shapes it hands back.
    #[test]
    fn sad_x4_matches_scalar() {
        let f = neon().sad_x4.expect("NEON installs the four-way SAD");
        let mut seed = 7u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 24) as u8
        };
        let (ss, rs) = (40usize, 72usize);
        let s: Vec<u8> = (0..ss * 64).map(|_| rnd()).collect();
        let r: Vec<u8> = (0..rs * 70).map(|_| rnd()).collect();
        for &(w, h) in &[(16, 16), (16, 8), (8, 16), (8, 8), (8, 4), (4, 8), (16, 64)] {
            let refs = [&r[0..], &r[3..], &r[rs + 1..], &r[5 * rs + 2..]];
            assert_eq!(
                f(&s, ss, refs, rs, w, h),
                crate::dsp::h264_enc::sad_x4_scalar(&s, ss, refs, rs, w, h),
                "{w}x{h}"
            );
        }
    }

    /// The four-way average-SATD against its definition: every shape the
    /// refinement scores (and those it hands back), pairs of two rows and
    /// of one row twice, random samples and the extremes.
    #[test]
    fn satd_avg_x4_matches_scalar() {
        let f = neon()
            .satd_avg_x4
            .expect("NEON installs the four-way average-SATD");
        let mut seed = 11u64;
        let rs = 72usize;
        for &(w, h) in &[(16, 16), (16, 8), (8, 16), (8, 8), (8, 4), (4, 8), (4, 4)] {
            for trial in 0..100 {
                let mk = |seed: &mut u64, n: usize| -> Vec<u8> {
                    (0..n)
                        .map(|_| match trial % 4 {
                            0 => 0,
                            1 => 255,
                            _ => lcg(seed) as u8,
                        })
                        .collect()
                };
                let s = mk(&mut seed, 40 * 20);
                let r = mk(&mut seed, rs * 24);
                let pairs = [
                    (&r[0..], &r[1..]),
                    (&r[rs + 2..], &r[rs + 2..]),
                    (&r[3 * rs..], &r[2 * rs + 5..]),
                    (&r[7..], &r[4 * rs + 3..]),
                ];
                assert_eq!(
                    f(&s, 40, pairs, rs, w, h),
                    crate::dsp::h264_enc::satd_avg_x4_scalar(&s, 40, pairs, rs, w, h),
                    "{w}x{h} trial {trial}"
                );
            }
        }
    }

    /// The four-way plane-sample SATD against its definition, and against
    /// the average-SATD of each view twice (what the refinement used for
    /// them before): every shape, random samples and the extremes.
    #[test]
    fn satd_x4_matches_scalar() {
        let d = neon();
        let f = d.satd_x4.expect("NEON installs the four-way SATD");
        let g = d
            .satd_avg_x4
            .expect("NEON installs the four-way average-SATD");
        let mut seed = 13u64;
        let rs = 72usize;
        for &(w, h) in &[(16, 16), (16, 8), (8, 16), (8, 8), (8, 4), (4, 8), (4, 4)] {
            for trial in 0..100 {
                let mk = |seed: &mut u64, n: usize| -> Vec<u8> {
                    (0..n)
                        .map(|_| match trial % 4 {
                            0 => 0,
                            1 => 255,
                            _ => lcg(seed) as u8,
                        })
                        .collect()
                };
                let s = mk(&mut seed, 40 * 20);
                let r = mk(&mut seed, rs * 24);
                let refs = [&r[0..], &r[rs + 2..], &r[3 * rs + 1..], &r[4 * rs + 7..]];
                let want = crate::dsp::h264_enc::satd_x4_scalar(&s, 40, refs, rs, w, h);
                assert_eq!(f(&s, 40, refs, rs, w, h), want, "{w}x{h} trial {trial}");
                assert_eq!(
                    g(&s, 40, refs.map(|v| (v, v)), rs, w, h),
                    want,
                    "{w}x{h} trial {trial}"
                );
            }
        }
    }

    /// The four-way bi-predicted SATD against its definition, as
    /// [`satd_avg_x4_matches_scalar`] with a held prediction beside.
    #[test]
    fn satd_bi_x4_matches_scalar() {
        let f = neon()
            .satd_bi_x4
            .expect("NEON installs the four-way bi-predicted SATD");
        let mut seed = 13u64;
        let rs = 72usize;
        for &(w, h) in &[(16, 16), (16, 8), (8, 16), (8, 8), (8, 4), (4, 8), (4, 4)] {
            for trial in 0..100 {
                let mk = |seed: &mut u64, n: usize| -> Vec<u8> {
                    (0..n)
                        .map(|_| match trial % 4 {
                            0 => 0,
                            1 => 255,
                            _ => lcg(seed) as u8,
                        })
                        .collect()
                };
                let s = mk(&mut seed, 40 * 20);
                let r = mk(&mut seed, rs * 24);
                let held = mk(&mut seed, 16 * 16);
                let pairs = [
                    (&r[0..], &r[1..]),
                    (&r[rs + 2..], &r[rs + 2..]),
                    (&r[3 * rs..], &r[2 * rs + 5..]),
                    (&r[7..], &r[4 * rs + 3..]),
                ];
                assert_eq!(
                    f(&s, 40, pairs, rs, &held, 16, w, h),
                    crate::dsp::h264_enc::satd_bi_x4_scalar(&s, 40, pairs, rs, &held, 16, w, h),
                    "{w}x{h} trial {trial}"
                );
            }
        }
    }

    /// The fused average-SATD against its definition, every block shape
    /// the search scores, random samples and the extremes.
    #[test]
    fn satd_avg_matches_scalar() {
        let f = neon().satd_avg.expect("NEON installs the fused SATD");
        let mut seed = 7u64;
        for &(w, h) in &[(16, 16), (16, 8), (8, 16), (8, 8), (8, 4), (4, 8), (4, 4)] {
            for trial in 0..200 {
                let mk = |seed: &mut u64| -> Vec<u8> {
                    (0..40 * 24)
                        .map(|_| match trial % 4 {
                            0 => 0,
                            1 => 255,
                            _ => lcg(seed) as u8,
                        })
                        .collect()
                };
                let (s, a, b) = (mk(&mut seed), mk(&mut seed), mk(&mut seed));
                let (ss, sa, sb) = (40, 37, 33);
                assert_eq!(
                    f(&s, ss, &a, sa, &b, sb, w, h),
                    crate::dsp::h264_enc::satd_avg_scalar(&s, ss, &a, sa, &b, sb, w, h),
                    "{w}x{h} trial {trial}"
                );
            }
        }
    }

    /// The fused difference and transform against its definition.
    #[test]
    fn sub_fdct_matches_scalar() {
        let d = neon();
        let (f4, f8) = (
            d.sub_fdct4.expect("NEON installs it"),
            d.sub_fdct8.expect("NEON installs it"),
        );
        let mut seed = 0x51u64;
        for trial in 0..400 {
            let mk = |seed: &mut u64| -> Vec<u8> {
                (0..24 * 12)
                    .map(|_| match trial % 4 {
                        0 => 0,
                        1 => 255,
                        _ => lcg(seed) as u8,
                    })
                    .collect()
            };
            let (s, p) = (mk(&mut seed), mk(&mut seed));
            let p = if trial % 8 == 1 {
                vec![0u8; p.len()]
            } else {
                p
            };
            let (mut a, mut b) = ([0i32; 16], [0i32; 16]);
            f4(&s, 24, &p, 19, &mut a);
            crate::dsp::h264_enc::sub_fdct4_scalar(&s, 24, &p, 19, &mut b);
            assert_eq!(a, b, "4x4 trial {trial}");
            let (mut a, mut b) = ([0i32; 64], [0i32; 64]);
            f8(&s, 24, &p, 19, &mut a);
            crate::dsp::h264_enc::sub_fdct8_scalar(&s, 24, &p, 19, &mut b);
            assert_eq!(a, b, "8x8 trial {trial}");
        }
    }

    #[test]
    fn transforms_match_scalar() {
        let s = H264EncDsp::SCALAR;
        let d = neon();
        let mut seed = 0x4dc7_u64;
        for bit_depth in [8u32, 9, 10, 12, 14] {
            let span = 1i32 << bit_depth;
            for round in 0..300 {
                let mut r = || match lcg(&mut seed) % 8 {
                    0 => 32767i16,
                    1 => -32768,
                    2 => lcg(&mut seed) as i16,
                    _ => ((lcg(&mut seed) as i32 % (2 * span - 1)) - (span - 1)) as i16,
                };
                let r4: [i16; 16] = std::array::from_fn(|_| r());
                let r8: [i16; 64] = std::array::from_fn(|_| r());
                let (mut w4, mut g4, mut w8, mut g8) =
                    ([0i32; 16], [0i32; 16], [0i32; 64], [0i32; 64]);
                (s.fdct4)(&r4, &mut w4);
                (d.fdct4)(&r4, &mut g4);
                (s.fdct8)(&r8, &mut w8);
                (d.fdct8)(&r8, &mut g8);
                assert_eq!(g4, w4, "fdct4, {bit_depth} bits, round {round}");
                assert_eq!(g8, w8, "fdct8, {bit_depth} bits, round {round}");
                let dc: [i32; 16] =
                    std::array::from_fn(|i| w4[i] * if round % 2 == 0 { 1 } else { 16 });
                let (mut wh, mut gh) = (dc, dc);
                (s.hadamard4)(&mut wh);
                (d.hadamard4)(&mut gh);
                assert_eq!(gh, wh, "hadamard4, round {round}");
            }
        }
    }

    #[test]
    fn quantisers_match_scalar() {
        let s = H264EncDsp::SCALAR;
        let d = neon();
        let mut seed = 0x9a41_u64;
        let flat = ScalingLists {
            list4x4: [[16; 16]; 6],
            list8x8: [[16; 64]; 6],
        };
        let small = ScalingLists {
            list4x4: [[4; 16]; 6],
            list8x8: [[4; 64]; 6],
        };
        for lists in [flat, small] {
            let q = Quant::new(&lists);
            for qp in 0..=87 {
                for intra in [true, false] {
                    let (qb4, qb8) = (qbits4(qp), qbits8(qp));
                    let (o4, o8) = (quant_offset(qb4, intra), quant_offset(qb8, intra));
                    let (mf4, mf8) = (&q.mf4[0][(qp % 6) as usize], &q.mf8[0][(qp % 6) as usize]);
                    for span in [255 * 36, 1023 * 64, 16383 * 256] {
                        let mut c = || match lcg(&mut seed) % 16 {
                            0 => 0,
                            1 => i32::MAX,
                            2 => i32::MIN + 1,
                            5 => lcg(&mut seed) as i32,
                            _ => (lcg(&mut seed) as i32 % (2 * span)) - span,
                        };
                        let c4: [i32; 16] = std::array::from_fn(|_| c());
                        let c8: [i32; 64] = std::array::from_fn(|_| c());
                        let (mut w4, mut g4, mut w8, mut g8) =
                            ([0i16; 16], [0i16; 16], [0i16; 64], [0i16; 64]);
                        let nw4 = (s.quant4)(&c4, &mut w4, mf4, qb4, o4);
                        let ng4 = (d.quant4)(&c4, &mut g4, mf4, qb4, o4);
                        let nw8 = (s.quant8)(&c8, &mut w8, mf8, qb8, o8);
                        let ng8 = (d.quant8)(&c8, &mut g8, mf8, qb8, o8);
                        assert_eq!(
                            (g4, ng4),
                            (w4, nw4),
                            "quant4 qp {qp} intra {intra} span {span}"
                        );
                        assert_eq!(
                            (g8, ng8),
                            (w8, nw8),
                            "quant8 qp {qp} intra {intra} span {span}"
                        );
                    }
                }
            }
        }
        // The calls handed back to the reference.
        let c: [i32; 16] = std::array::from_fn(|i| i as i32 * 977 - 7000);
        let mut neg = [13107i32; 16];
        neg[5] = -3;
        for (mf, qb, off) in [(neg, 15, 100), ([13107; 16], 15, -5), ([13107; 16], 63, 0)] {
            let (mut want, mut got) = ([0i16; 16], [0i16; 16]);
            let nw = (s.quant4)(&c, &mut want, &mf, qb, off);
            let ng = (d.quant4)(&c, &mut got, &mf, qb, off);
            assert_eq!((got, ng), (want, nw), "fallback qbits {qb} offset {off}");
        }
    }
}
