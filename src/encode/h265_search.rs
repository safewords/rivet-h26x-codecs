//! Motion-search helpers for the H.265 encoder: the half-sample planes a
//! reference picture is searched through, and the price of a motion
//! vector difference.
//!
//! # Half-sample planes
//!
//! The sub-sample refinement used to interpolate every candidate it
//! scored: eight half-sample and eight quarter-sample neighbours per
//! prediction unit per reference, each a full 8-tap filter pass over the
//! block — a quarter of an encode's CPU went to `qpel_h` / `qpel_v2`
//! alone. A reference picture is searched by every unit of every picture
//! that predicts from it, so its three half-sample phases are computed
//! once, when it becomes a reference, and a half-sample candidate is then
//! a read.
//!
//! The planes are **exact**: each is what 8.5.3.3.3.1 makes of the
//! reference at that phase for a uni-predicted block under default
//! weighting — the 8-tap half-sample filter, the intermediate shifts
//! (`shift1` = `Min(4, BitDepth - 8)`, then 6 for the second pass) and
//! the final `(x + offset) >> (14 - BitDepth)` rounding and clip. So the
//! SATD of a half-sample candidate read from them equals the SATD of the
//! prediction the decoder would make, and the search compares candidates
//! on the same numbers as before. A unit under explicit weighting scores
//! through the interpolating path as it always did.
//!
//! The planes cover the picture and [`REACH`] samples of its padding on
//! every side; a candidate reaching further is interpolated when scored.

use crate::dsp::hevc::HevcDsp;
use crate::hevc::frame::Plane16;
use crate::sample::Sample;

/// Above this many luma samples a reference keeps its three half-sample
/// phases only: fifteen phases of a 3840x2160 picture are 140 MB.
const QUARTER_LIMIT: usize = 2_200_000;

/// The fractional-sample phases of one reference luma plane, laid out
/// exactly as that plane is (same stride, same padding, same origin), so
/// an offset computed for the reference indexes these too. Phase
/// `(fx, fy)` (quarter samples) lives at `planes[fy * 4 + fx - 1]`; a
/// plane is empty when the phase was not built.
pub(crate) struct HalfPlanes<S: Sample> {
    /// The fifteen fractional phases, `(0, 0)` excluded.
    pub planes: Vec<Vec<S>>,
    /// The reference plane's stride.
    pub stride: usize,
    /// The reference plane's padding.
    pub pad: usize,
    /// How far outside the picture the phases were built: a block
    /// reaching further is interpolated as it is scored.
    pub ext: usize,
    /// Picture width and height.
    pub width: usize,
    /// See `width`.
    pub height: usize,
    /// The address of the reference plane's samples these were built
    /// from: how a scoring call that holds only the plane finds them
    /// (`InterPicture::planes_of`). A plane's buffer does not move while
    /// its frame is a reference.
    pub source: usize,
}

/// How far inside the padded area the planes are valid.
pub(crate) const MARGIN: usize = 4;

/// How far outside the picture the phases are built. Vectors reaching
/// further are rare, and are scored by interpolating as they always could
/// be: the planes are a cache of exact values, so where they stop changes
/// no decision. Built over the whole padded area they cost a CIF picture
/// more than twice its own area.
const REACH: usize = 16;

impl<S: Sample> HalfPlanes<S> {
    /// Build the fractional phases of `p`, whose border must already be
    /// extended, at `bit_depth`: all fifteen, or the three half-sample
    /// ones above [`QUARTER_LIMIT`] luma samples. Through the decoder's
    /// own interpolation kernels (`dsp`), 64x64 tiles at a time, over the
    /// padded area less [`MARGIN`].
    pub fn new(p: &Plane16<S>, bit_depth: u32, dsp: &HevcDsp<S>) -> Self {
        const T: usize = 64;
        let (stride, rows) = (p.stride, p.data.len() / p.stride);
        let quarter = p.width * p.height <= QUARTER_LIMIT;
        let shift1 = bit_depth.min(12) as i32 - 8;
        let shift3 = 14 - bit_depth as i32;
        let max = (1i32 << bit_depth) - 1;
        let wanted =
            |fx: usize, fy: usize| quarter || (fx.is_multiple_of(2) && fy.is_multiple_of(2));
        let d = &p.data;
        let mut planes: Vec<Vec<S>> = (1..16)
            .map(|ph| {
                if wanted(ph % 4, ph / 4) {
                    vec![S::default(); d.len()]
                } else {
                    Vec::new()
                }
            })
            .collect();
        let mut tmp = vec![0i16; T * (T + 7)];
        let mut out14 = vec![0i16; T * T];
        let ext = REACH.min(p.pad - MARGIN);
        let (lo, hi_x, hi_y) = (
            p.pad - ext,
            p.pad + p.width + ext,
            (p.pad + p.height + ext).min(rows - MARGIN),
        );
        let mut y0 = lo;
        while y0 < hi_y {
            let th = T.min(hi_y - y0);
            let mut x0 = lo;
            while x0 < hi_x {
                let tw = T.min(hi_x - x0);
                let at = y0 * stride + x0;
                for fx in 0..4usize {
                    // The horizontal pass over th + 7 rows, from three rows
                    // above, for the phases that interpolate both ways.
                    if fx != 0 && (1..4).any(|fy| wanted(fx, fy)) {
                        (dsp.qpel_h)(
                            &mut tmp,
                            &d[at - 3 * stride - 3..],
                            stride,
                            tw,
                            th + 7,
                            fx,
                            shift1,
                        );
                    }
                    for fy in 0..4usize {
                        if (fx, fy) == (0, 0) || !wanted(fx, fy) {
                            continue;
                        }
                        match (fx, fy) {
                            (_, 0) => {
                                (dsp.qpel_h)(&mut out14, &d[at - 3..], stride, tw, th, fx, shift1)
                            }
                            (0, _) => (dsp.qpel_v)(
                                &mut out14,
                                &d[at - 3 * stride..],
                                stride,
                                tw,
                                th,
                                fy,
                                shift1,
                            ),
                            _ => (dsp.qpel_v2)(&mut out14, &tmp, tw, tw, th, fy),
                        }
                        (dsp.uni)(
                            &mut planes[fy * 4 + fx - 1][at..],
                            stride,
                            &out14,
                            tw,
                            th,
                            shift3,
                            max,
                        );
                    }
                }
                x0 += tw;
            }
            y0 += th;
        }
        HalfPlanes {
            planes,
            stride,
            pad: p.pad,
            ext,
            width: p.width,
            height: p.height,
            source: p.data.as_ptr() as usize,
        }
    }

    /// The plane and offset of the fractional position `mv` (quarter-sample
    /// units) for a block at luma `(x, y)` of `w` by `h`, or `None` when the
    /// block reaches past the valid area, `mv` is on the full-sample grid
    /// (the caller reads the reference itself) or its phase was not built.
    #[allow(clippy::too_many_arguments)]
    pub fn at(
        &self,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        mvx: i32,
        mvy: i32,
    ) -> Option<(&[S], usize)> {
        let phase = ((mvy & 3) * 4 + (mvx & 3)) as usize;
        if phase == 0 || self.planes[phase - 1].is_empty() {
            return None;
        }
        let xi = x as i32 + (mvx >> 2);
        let yi = y as i32 + (mvy >> 2);
        let lim = self.ext as i32;
        if xi < -lim
            || yi < -lim
            || xi + w as i32 > self.width as i32 + lim
            || yi + h as i32 > self.height as i32 + lim
        {
            return None;
        }
        let off = ((self.pad as i32 + yi) as usize) * self.stride + (self.pad as i32 + xi) as usize;
        Some((&self.planes[phase - 1][off..], self.stride))
    }
}

/// Approximate bits of one motion vector difference component `d`
/// (quarter samples) as `write_mvd` codes it: `abs_mvd_greater0_flag`,
/// `abs_mvd_greater1_flag`, the first-order Exp-Golomb remainder
/// `abs_mvd_minus2` and the sign, each context-coded flag priced at one
/// bit. Used inside the search, where a table of exact prices per
/// candidate would cost more than it changes.
#[inline]
pub(crate) fn mvd_component_bits(d: i32) -> u32 {
    let a = d.unsigned_abs();
    match MVD_BITS.get(a as usize) {
        Some(&b) => u32::from(b),
        None => mvd_bits_of(a),
    }
}

/// [`mvd_component_bits`] for the magnitudes a search meets most, by table.
static MVD_BITS: [u8; 2048] = {
    let mut t = [0u8; 2048];
    let mut a = 0;
    while a < 2048 {
        t[a] = mvd_bits_of(a as u32) as u8;
        a += 1;
    }
    t
};

/// [`mvd_component_bits`] of magnitude `a`, computed.
const fn mvd_bits_of(a: u32) -> u32 {
    match a {
        0 => 1,
        1 => 3,
        _ => {
            // First-order Exp-Golomb of a - 2 (9.3.3.6).
            let (mut v, mut k, mut bits) = (a - 2, 1u32, 0u32);
            while v >= 1 << k {
                bits += 1;
                v -= 1 << k;
                k += 1;
            }
            3 + bits + 1 + k
        }
    }
}

/// Approximate bits of the difference between `mv` and the cheaper of the
/// two predictors, plus the predictor flag.
#[inline]
pub(crate) fn mv_bits(mvx: i32, mvy: i32, mvp: &[(i32, i32); 2]) -> u32 {
    let b = |p: (i32, i32)| mvd_component_bits(mvx - p.0) + mvd_component_bits(mvy - p.1);
    b(mvp[0]).min(b(mvp[1])) + 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cabac_enc::CabacEncoder;
    use crate::hevc::ctu::write_mvd;
    use crate::hevc::ctx::Contexts;
    use crate::hevc::frame::Mv;

    /// The approximate price stays within two bits of what the writer
    /// spends on a lone vector difference at the slice's initial
    /// probabilities, across the range a search reaches.
    #[test]
    fn the_mvd_price_tracks_the_writer() {
        for d in [
            -300i32, -64, -17, -5, -2, -1, 0, 1, 2, 3, 4, 7, 8, 33, 100, 512,
        ] {
            let mut cx = Contexts::new(1, 30);
            let mut e = CabacEncoder::counting();
            write_mvd(&mut e, &mut cx, Mv::new(d as i16, 0));
            let real = e.fractional_bits();
            let ours = f64::from(mvd_component_bits(d) + mvd_component_bits(0));
            assert!(
                (real - ours).abs() <= 2.5,
                "d {d}: writer {real:.2}, model {ours}"
            );
        }
    }

    /// The fractional planes are the decoder's own interpolation: every
    /// phase at every position equals `predict_block`'s output for a
    /// uni-predicted block under default weighting.
    #[test]
    fn half_planes_equal_the_decoders_interpolation() {
        use crate::dsp::Cpu;
        use crate::dsp::hevc::HevcDsp;
        use crate::hevc::frame::Frame;
        use crate::picture::ChromaFormat;
        let (w, h) = (40usize, 24usize);
        let mut f = Frame::<u8>::new(w, h, ChromaFormat::Yuv420, 8);
        let mut s = 7u32;
        for y in 0..h {
            for x in 0..w {
                s = s.wrapping_mul(1103515245).wrapping_add(12345);
                let o = f.y.offset(x as isize, y as isize);
                f.y.data[o] = ((s >> 16) & 255) as u8;
            }
        }
        f.extend_rows(0, h);
        let dsp = HevcDsp::<u8>::new(Cpu::SCALAR);
        let hp = HalfPlanes::new(&f.y, 8, &dsp);
        let mut out14 = vec![0i16; 64 * 64];
        let mut tmp = vec![0i16; 64 * 80];
        let mut out = vec![0u8; 64 * 64];
        for &(mvx, mvy) in &[
            (2i32, 0i32),
            (0, 2),
            (2, 2),
            (-6, 10),
            (14, -2),
            (-30, -30),
            (1, 0),
            (0, 3),
            (1, 1),
            (3, 2),
            (-7, 5),
            (13, -11),
        ] {
            for &(x, y) in &[(0usize, 0usize), (8, 4), (32, 16)] {
                let (bw, bh) = (8usize, 8usize);
                let xi = x as i32 + (mvx >> 2);
                let yi = y as i32 + (mvy >> 2);
                let (fx, fy) = ((mvx & 3) as usize, (mvy & 3) as usize);
                let base = f.y.offset((xi - 3) as isize, (yi - 3) as isize);
                let win = &f.y.data[base..];
                let st = f.y.stride;
                match (fx, fy) {
                    (_, 0) => (dsp.qpel_h)(&mut out14, &win[3 * st..], st, bw, bh, fx, 0),
                    (0, _) => (dsp.qpel_v)(&mut out14, &win[3..], st, bw, bh, fy, 0),
                    _ => {
                        (dsp.qpel_h)(&mut tmp, win, st, bw, bh + 7, fx, 0);
                        (dsp.qpel_v2)(&mut out14, &tmp, bw, bw, bh, fy);
                    }
                }
                (dsp.uni)(&mut out, bw, &out14, bw, bh, 6, 255);
                let (p, ps) = hp.at(x, y, bw, bh, mvx, mvy).expect("inside");
                for yy in 0..bh {
                    for xx in 0..bw {
                        assert_eq!(
                            p[yy * ps + xx],
                            out[yy * bw + xx],
                            "mv ({mvx},{mvy}) at ({x},{y})"
                        );
                    }
                }
            }
        }
    }
}
