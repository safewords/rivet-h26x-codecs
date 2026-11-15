//! Half-sample planes and the motion search that reads them.
//!
//! H.264's luma interpolation (8.4.2.2.1) has a property an encoder can
//! lean on: every quarter-sample position is the rounded average
//! `(a + b + 1) >> 1` of two samples drawn from four planes — the integer
//! samples `G`, the horizontal half samples `b`, the vertical half samples
//! `h` and the centre half samples `j`. So once the three half-sample
//! planes of a reference are built (one six-tap pass each, once per
//! reference picture), *any* candidate vector is a pointer into a plane or
//! one averaging pass over two — no six-tap filter in the search loop.
//! The planes are built with the standard's own arithmetic, so a block
//! fetched here is bit for bit the decoder's prediction for the same
//! vector; the test at the bottom holds that against the decoder's
//! quarter-sample kernels at all sixteen positions.
//!
//! The search itself is the textbook one: candidate predictors scored by
//! SAD plus the multiplier times the vector's rate, a hexagon descent and
//! a square refinement at whole samples, then square refinements at half
//! and quarter samples scored by SATD plus rate.

use crate::dsp::distortion::DistortionDsp;
use crate::encode::h264_syntax::Recon;
use crate::h264::frame::Mv;
use crate::sample::Sample;

/// The three half-sample planes of one reference luma plane, at the same
/// geometry (stride, border) as the plane itself: `h` holds the sample
/// half a step right of each integer position (`b`), `v` half a step
/// below (`h` in the standard's naming) and `c` both (`j`).
pub(crate) struct Hpel<S: Sample> {
    pub(crate) h: Recon<S>,
    pub(crate) v: Recon<S>,
    pub(crate) c: Recon<S>,
}

#[inline(always)]
fn tap6(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32) -> i32 {
    a + f - 5 * (b + e) + 20 * (c + d)
}

/// Build the half-sample planes of `r`, whose border is already
/// replicated. Positions whose six-tap window would leave the padded plane
/// copy the integer sample instead; the search never reads them (its
/// window keeps every read inside, see [`search_bounds`]).
pub(crate) fn build_hpel<S: Sample>(r: &Recon<S>, max: i32) -> Hpel<S> {
    let stride = r.stride;
    let rows = r.data.len() / stride;
    let mut h = r.clone();
    let mut v = r.clone();
    let mut c = r.clone();
    let d = &r.data;
    // Horizontal half samples, every row.
    for y in 0..rows {
        let row = &d[y * stride..(y + 1) * stride];
        let out = &mut h.data[y * stride..(y + 1) * stride];
        for x in 2..stride - 3 {
            let t = tap6(
                row[x - 2].to_i32(),
                row[x - 1].to_i32(),
                row[x].to_i32(),
                row[x + 1].to_i32(),
                row[x + 2].to_i32(),
                row[x + 3].to_i32(),
            );
            out[x] = S::from_i32(((t + 16) >> 5).clamp(0, max));
        }
    }
    // Vertical half samples and, from their unclipped intermediates, the
    // centre ones (j1 is the horizontal six-tap over the vertical
    // intermediates, 8.4.2.2.1).
    let mut inter = vec![0i32; stride];
    for y in 2..rows.saturating_sub(3) {
        for (x, t) in inter.iter_mut().enumerate() {
            *t = tap6(
                d[(y - 2) * stride + x].to_i32(),
                d[(y - 1) * stride + x].to_i32(),
                d[y * stride + x].to_i32(),
                d[(y + 1) * stride + x].to_i32(),
                d[(y + 2) * stride + x].to_i32(),
                d[(y + 3) * stride + x].to_i32(),
            );
        }
        let vo = &mut v.data[y * stride..(y + 1) * stride];
        for x in 0..stride {
            vo[x] = S::from_i32(((inter[x] + 16) >> 5).clamp(0, max));
        }
        let co = &mut c.data[y * stride..(y + 1) * stride];
        for x in 2..stride - 3 {
            let t = tap6(
                inter[x - 2],
                inter[x - 1],
                inter[x],
                inter[x + 1],
                inter[x + 2],
                inter[x + 3],
            );
            co[x] = S::from_i32(((t + 512) >> 10).clamp(0, max));
        }
    }
    Hpel { h, v, c }
}

/// A reference as the search reads it: the integer plane and its three
/// half-sample planes.
pub(crate) struct MeRef<'a, S: Sample> {
    /// `[G, b, h, j]`: integer, horizontal half, vertical half, centre.
    pub(crate) p: [&'a Recon<S>; 4],
}

/// For each quarter-sample fraction `(fy * 4 + fx)`: the one or two
/// `(plane, dx, dy)` sources whose rounded average is the sample
/// (8.4.2.2.1's equations for a, c, d, n, f, i, k, q, e, g, p, r). A second
/// plane of 255 means the position is a plane sample itself.
const QPEL: [(u8, i8, i8, u8, i8, i8); 16] = [
    (0, 0, 0, 255, 0, 0), // G
    (0, 0, 0, 1, 0, 0),   // a = (G + b)
    (1, 0, 0, 255, 0, 0), // b
    (1, 0, 0, 0, 1, 0),   // c = (H + b)
    (0, 0, 0, 2, 0, 0),   // d = (G + h)
    (1, 0, 0, 2, 0, 0),   // e = (b + h)
    (1, 0, 0, 3, 0, 0),   // f = (b + j)
    (1, 0, 0, 2, 1, 0),   // g = (b + m)
    (2, 0, 0, 255, 0, 0), // h
    (2, 0, 0, 3, 0, 0),   // i = (h + j)
    (3, 0, 0, 255, 0, 0), // j
    (3, 0, 0, 2, 1, 0),   // k = (j + m)
    (2, 0, 0, 0, 0, 1),   // n = (M + h)
    (2, 0, 0, 1, 0, 1),   // p = (h + s)
    (3, 0, 0, 1, 0, 1),   // q = (j + s)
    (2, 1, 0, 1, 0, 1),   // r = (m + s)
];

impl<'a, S: Sample> MeRef<'a, S> {
    #[inline(always)]
    fn at(&self, plane: u8, x: i32, y: i32) -> &'a [S] {
        let r = self.p[plane as usize];
        &r.data[r.offset(x as isize, y as isize)..]
    }

    /// The `w` by `h` luma prediction of the block at picture position
    /// `(x, y)` for vector `mv`, as `(samples, stride)`: a view straight
    /// into a plane where the position is one, else the average of two
    /// views written into `buf` (stride 16).
    #[inline]
    pub(crate) fn pred<'b>(
        &self,
        x: i32,
        y: i32,
        mv: Mv,
        w: usize,
        h: usize,
        buf: &'b mut [S; 256],
    ) -> (&'b [S], usize)
    where
        'a: 'b,
    {
        let (xi, yi) = (x + (mv.x as i32 >> 2), y + (mv.y as i32 >> 2));
        let (p1, dx1, dy1, p2, dx2, dy2) = QPEL[((mv.y & 3) * 4 + (mv.x & 3)) as usize];
        let a = self.at(p1, xi + dx1 as i32, yi + dy1 as i32);
        let stride = self.p[0].stride;
        if p2 == 255 {
            return (a, stride);
        }
        let b = self.at(p2, xi + dx2 as i32, yi + dy2 as i32);
        avg_into(buf, 16, a, stride, b, stride, w, h);
        (&buf[..], 16)
    }
}

/// `(a + b + 1) >> 1` over a `w` by `h` block.
/// [`avg_into`]'s 8-bit rows at a width known when compiled: the bounds
/// checked once for the whole block, then each row one vector average.
#[inline(always)]
fn avg_rows_u8<const W: usize>(
    out: &mut [u8],
    os: usize,
    a: &[u8],
    as_: usize,
    b: &[u8],
    bs: usize,
    h: usize,
) {
    if h == 0 {
        return;
    }
    assert!(
        a.len() >= (h - 1) * as_ + W && b.len() >= (h - 1) * bs + W && out.len() >= (h - 1) * os + W
    );
    for y in 0..h {
        // SAFETY: the assertion above covers row `y`'s `W` samples of all
        // three blocks.
        unsafe {
            let ra = &*(a.as_ptr().add(y * as_) as *const [u8; W]);
            let rb = &*(b.as_ptr().add(y * bs) as *const [u8; W]);
            let ro = &mut *(out.as_mut_ptr().add(y * os) as *mut [u8; W]);
            for x in 0..W {
                ro[x] = ((ra[x] as u16 + rb[x] as u16 + 1) >> 1) as u8;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn avg_into<S: Sample>(
    out: &mut [S],
    os: usize,
    a: &[S],
    as_: usize,
    b: &[S],
    bs: usize,
    w: usize,
    h: usize,
) {
    if S::BYTES == 1 {
        // SAFETY: a one-byte `Sample` is `u8` (the trait has two
        // implementors, `u8` and `u16`), so the slices are byte slices of
        // the same lengths.
        let (out, a, b) = unsafe {
            (
                std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out.len()),
                std::slice::from_raw_parts(a.as_ptr() as *const u8, a.len()),
                std::slice::from_raw_parts(b.as_ptr() as *const u8, b.len()),
            )
        };
        // Fixed widths, so each row is one vector rounding-average (the
        // compiler's `urhadd` / `pavgb`) rather than a byte loop.
        match w {
            16 => avg_rows_u8::<16>(out, os, a, as_, b, bs, h),
            8 => avg_rows_u8::<8>(out, os, a, as_, b, bs, h),
            4 => avg_rows_u8::<4>(out, os, a, as_, b, bs, h),
            _ => {
                for y in 0..h {
                    let (ra, rb) = (&a[y * as_..y * as_ + w], &b[y * bs..y * bs + w]);
                    let ro = &mut out[y * os..y * os + w];
                    for ((o, &p), &q) in ro.iter_mut().zip(ra).zip(rb) {
                        *o = ((p as u16 + q as u16 + 1) >> 1) as u8;
                    }
                }
            }
        }
        return;
    }
    for y in 0..h {
        let (ra, rb) = (&a[y * as_..y * as_ + w], &b[y * bs..y * bs + w]);
        let ro = &mut out[y * os..y * os + w];
        for x in 0..w {
            ro[x] = S::from_i32((ra[x].to_i32() + rb[x].to_i32() + 1) >> 1);
        }
    }
}

/// The bits CABAC spends on one `mvd` component of value `d` (quarter
/// samples), estimated from its binarisation (9.3.2.3, UEG3 with a
/// truncated-unary prefix of up to nine context-coded bins, a third-order
/// Exp-Golomb suffix and a sign): the prefix bins are charged a little
/// under a bit each, since their contexts learn, the bypass bins one bit
/// each.
pub(crate) fn mvd_bits(d: i32) -> f32 {
    let a = d.unsigned_abs();
    if a == 0 {
        return 1.0;
    }
    let prefix = a.min(9) as f32;
    let mut bits = 1.0 + 0.85 * prefix; // first bin + the rest of the prefix
    if a < 9 {
        bits += 0.85; // the terminating zero of the unary prefix
    } else {
        // EG3 of a - 9: a unary count of k past 3, then k suffix bits.
        let mut v = a - 9;
        let mut k = 3u32;
        while v >= (1 << k) {
            v -= 1 << k;
            k += 1;
        }
        bits += (k - 3 + 1 + k) as f32;
    }
    bits + 1.0 // sign
}

/// Half the span of the vector-rate table, in quarter samples.
pub(crate) const MV_SPAN: i32 = 4096;

/// The rate side of a vector's cost at one multiplier: `lambda` times
/// [`mvd_bits`], tabulated over `-MV_SPAN..=MV_SPAN` in whole units.
pub(crate) struct MvCostTable {
    tab: Vec<u16>,
}

impl MvCostTable {
    pub(crate) fn new(lambda: f32) -> Self {
        let tab = (-MV_SPAN..=MV_SPAN)
            .map(|d| (lambda * mvd_bits(d) + 0.5).min(65535.0) as u16)
            .collect();
        MvCostTable { tab }
    }

    /// The cost of differing from the predictor by `d` in one component.
    #[inline(always)]
    pub(crate) fn comp(&self, d: i32) -> u32 {
        self.tab[(d.clamp(-MV_SPAN, MV_SPAN) + MV_SPAN) as usize] as u32
    }

    /// The cost of `mv` against predictor `p`.
    #[inline(always)]
    pub(crate) fn of(&self, mv: Mv, p: Mv) -> u32 {
        self.comp(mv.x as i32 - p.x as i32) + self.comp(mv.y as i32 - p.y as i32)
    }
}

/// The search's whole-sample window for a `w` by `h` block at `(x, y)` in
/// a plane of `r`'s geometry: every vector whose six-tap window, and that
/// of every refinement reachable from it, reads inside the padded plane,
/// held to the vertical range `vr` (quarter samples converted by the
/// caller). As `(lo_x, hi_x, lo_y, hi_y)` in whole samples.
pub(crate) fn search_bounds<S: Sample>(
    r: &Recon<S>,
    x: i32,
    y: i32,
    w: usize,
    h: usize,
    vr: i32,
) -> (i32, i32, i32, i32) {
    let pad = r.pad as i32;
    let (lo_x, hi_x) = (4 - pad - x, r.width as i32 + pad - (w as i32 + 4) - x);
    let (lo_y, hi_y) = (4 - pad - y, r.height as i32 + pad - (h as i32 + 4) - y);
    (lo_x, hi_x, lo_y.max(-vr), hi_y.min(vr))
}

/// How hard one search tries.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MeCfg {
    /// Whole-sample range of the descent around its best start.
    pub(crate) range: i32,
    /// Half-sample refinement rounds.
    pub(crate) hpel_iters: u8,
    /// Quarter-sample refinement rounds.
    pub(crate) qpel_iters: u8,
    /// Uneven multi-hexagon: a cross and rings of hexagons around the
    /// start before the descent, for motion the predictors miss.
    pub(crate) umh: bool,
    /// Refine quarter samples on the four-point diamond rather than the
    /// eight-point square.
    pub(crate) qpel_diamond: bool,
    /// Small-diamond descent instead of the hexagon (cheapest).
    pub(crate) dia: bool,
}

/// One block's search: everything fixed while candidates are scored.
pub(crate) struct Search<'a, 'b, S: Sample> {
    pub(crate) dist: &'b DistortionDsp<S>,
    pub(crate) r: &'b MeRef<'a, S>,
    /// The source block (its top-left sample) and stride.
    pub(crate) src: &'b [S],
    pub(crate) src_stride: usize,
    /// The block's picture position and size.
    pub(crate) x: i32,
    pub(crate) y: i32,
    pub(crate) w: usize,
    pub(crate) h: usize,
    /// The vector predictor the rate is measured from.
    pub(crate) pred: Mv,
    pub(crate) cost: &'b MvCostTable,
    /// Whole-sample window, see [`search_bounds`].
    pub(crate) bounds: (i32, i32, i32, i32),
}

impl<S: Sample> Search<'_, '_, S> {
    #[inline]
    fn sad_full(&self, fx: i32, fy: i32) -> u32 {
        let r = self.r.p[0];
        let off = r.offset((self.x + fx) as isize, (self.y + fy) as isize);
        (self.dist.sad)(
            self.src,
            self.src_stride,
            &r.data[off..],
            r.stride,
            self.w,
            self.h,
        ) + self
            .cost
            .of(Mv::new((fx * 4) as i16, (fy * 4) as i16), self.pred)
    }

    #[inline]
    fn inside(&self, fx: i32, fy: i32) -> bool {
        let (lx, hx, ly, hy) = self.bounds;
        fx >= lx && fx <= hx && fy >= ly && fy <= hy
    }

    /// SATD of the prediction at `mv` (no rate).
    #[inline]
    pub(crate) fn satd(&self, mv: Mv, buf: &mut [S; 256]) -> u32 {
        let (p, ps) = self.r.pred(self.x, self.y, mv, self.w, self.h, buf);
        (self.dist.satd)(self.src, self.src_stride, p, ps, self.w, self.h)
    }

    /// The vector in quarter samples is inside the window.
    #[inline]
    pub(crate) fn inside_q(&self, mv: Mv) -> bool {
        let (lx, hx, ly, hy) = self.bounds;
        let (x, y) = (mv.x as i32, mv.y as i32);
        x >= lx * 4 && x <= hx * 4 && y >= ly * 4 && y <= hy * 4
    }

    /// Run the search from `cands` (quarter-sample vectors; rounded to
    /// whole samples for the descent). Returns the vector, its SATD and
    /// its full cost (SATD plus rate).
    pub(crate) fn run(&self, cands: &[Mv], cfg: &MeCfg) -> (Mv, u32, u32) {
        let (lx, hx, ly, hy) = self.bounds;
        let clampf = |m: Mv| {
            (
                ((m.x as i32 + 2) >> 2).clamp(lx, hx),
                ((m.y as i32 + 2) >> 2).clamp(ly, hy),
            )
        };
        let mut best = clampf(self.pred);
        let mut bcost = self.sad_full(best.0, best.1);
        let mut tried = [(i32::MIN, i32::MIN); 12];
        let mut nt = 0;
        for &c in cands {
            let f = clampf(c);
            if f == best || tried[..nt].contains(&f) {
                continue;
            }
            if nt < tried.len() {
                tried[nt] = f;
                nt += 1;
            }
            let s = self.sad_full(f.0, f.1);
            if s < bcost {
                bcost = s;
                best = f;
            }
        }
        let start = best;
        let (bx0, bx1) = ((start.0 - cfg.range).max(lx), (start.0 + cfg.range).min(hx));
        let (by0, by1) = ((start.1 - cfg.range).max(ly), (start.1 + cfg.range).min(hy));
        let ok = |fx: i32, fy: i32| fx >= bx0 && fx <= bx1 && fy >= by0 && fy <= by1;
        let try_at = |fx: i32, fy: i32, best: &mut (i32, i32), bcost: &mut u32| {
            if ok(fx, fy) && (fx, fy) != *best {
                let s = self.sad_full(fx, fy);
                if s < *bcost {
                    *bcost = s;
                    *best = (fx, fy);
                }
            }
        };
        if cfg.umh {
            // A cross out to the range, then hexagon rings at growing
            // radii — cheap coverage for motion no predictor saw.
            let c = best;
            let mut r = 2;
            while r <= cfg.range {
                for (dx, dy) in [(-r, 0), (r, 0), (0, -r), (0, r)] {
                    try_at(c.0 + dx, c.1 + dy, &mut best, &mut bcost);
                }
                r += 2;
            }
            let mut r = 4;
            while r <= cfg.range {
                for (dx, dy) in [
                    (-2, 0),
                    (-2, 1),
                    (-2, -1),
                    (2, 0),
                    (2, 1),
                    (2, -1),
                    (-1, 2),
                    (1, 2),
                    (0, 2),
                    (-1, -2),
                    (1, -2),
                    (0, -2),
                    (-2, 2),
                    (2, 2),
                    (-2, -2),
                    (2, -2),
                ] {
                    try_at(c.0 + dx * r / 2, c.1 + dy * r / 2, &mut best, &mut bcost);
                }
                r += 4;
            }
        }
        if cfg.dia {
            // Small diamond to a local minimum.
            for _ in 0..cfg.range {
                let c = best;
                for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                    try_at(c.0 + dx, c.1 + dy, &mut best, &mut bcost);
                }
                if best == c {
                    break;
                }
            }
            return self.refine(Mv::new((best.0 * 4) as i16, (best.1 * 4) as i16), cfg);
        }
        // Hexagon descent. After the first ring, a move to one of the six
        // vertices leaves three of the new ring's points already scored
        // (they were the old ring's), so only the other three are tried.
        const HEX: [(i32, i32); 6] = [(-2, 0), (-1, -2), (1, -2), (2, 0), (1, 2), (-1, 2)];
        let c = best;
        let mut dir = usize::MAX;
        for (k, &(dx, dy)) in HEX.iter().enumerate() {
            let before = best;
            try_at(c.0 + dx, c.1 + dy, &mut best, &mut bcost);
            if best != before {
                dir = k;
            }
        }
        let mut it = 1;
        while dir != usize::MAX && it < cfg.range {
            let c = best;
            let mut moved = usize::MAX;
            for k in [dir + 5, dir, dir + 1] {
                let k = k % 6;
                let before = best;
                try_at(c.0 + HEX[k].0, c.1 + HEX[k].1, &mut best, &mut bcost);
                if best != before {
                    moved = k;
                }
            }
            dir = moved;
            it += 1;
        }
        // Square refinement.
        let c = best;
        for (dx, dy) in [
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        ] {
            try_at(c.0 + dx, c.1 + dy, &mut best, &mut bcost);
        }
        debug_assert!(self.inside(best.0, best.1));
        self.refine(Mv::new((best.0 * 4) as i16, (best.1 * 4) as i16), cfg)
    }

    /// Sub-sample refinement from `mv`: square rings at half, then quarter
    /// samples, scored by SATD plus rate. Returns `(mv, satd, cost)`.
    pub(crate) fn refine(&self, mv: Mv, cfg: &MeCfg) -> (Mv, u32, u32) {
        let mut buf = [S::default(); 256];
        let mut best = mv;
        let mut bsatd = self.satd(best, &mut buf);
        let mut bcost = bsatd + self.cost.of(best, self.pred);
        const RING: [(i16, i16); 8] = [
            (0, -1),
            (-1, 0),
            (1, 0),
            (0, 1),
            (-1, -1),
            (1, -1),
            (-1, 1),
            (1, 1),
        ];
        for (step, iters) in [(2i16, cfg.hpel_iters), (1, cfg.qpel_iters)] {
            // Quarter samples: the diamond, when asked.
            let pts = if step == 1 && cfg.qpel_diamond { 4 } else { 8 };
            for _ in 0..iters {
                let c = best;
                for &(dx, dy) in &RING[..pts] {
                    let m = Mv::new(c.x + dx * step, c.y + dy * step);
                    if !self.inside_q(m) {
                        continue;
                    }
                    let rc = self.cost.of(m, self.pred);
                    if rc >= bcost {
                        continue;
                    }
                    let s = self.satd(m, &mut buf);
                    if s + rc < bcost {
                        bcost = s + rc;
                        bsatd = s;
                        best = m;
                    }
                }
                if best == c {
                    break;
                }
            }
        }
        (best, bsatd, bcost)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::Cpu;
    use crate::dsp::h264::{H264Dsp, PRED_STRIDE};

    /// Every quarter-sample position fetched through the planes equals the
    /// decoder's own quarter-sample kernel at the same vector.
    #[test]
    fn plane_predictions_are_the_decoders() {
        let (w, h, pad) = (48usize, 40usize, 32usize);
        let mut r = Recon::<u8>::new(w, h, pad);
        let mut s = 12345u32;
        for y in 0..h {
            for x in 0..w {
                s = s.wrapping_mul(1103515245).wrapping_add(12345);
                let o = r.offset(x as isize, y as isize);
                r.data[o] = (s >> 16) as u8;
            }
        }
        r.extend_edges(false);
        let hp = build_hpel(&r, 255);
        let me = MeRef {
            p: [&r, &hp.h, &hp.v, &hp.c],
        };
        let dsp = H264Dsp::<u8>::new(Cpu::SCALAR);
        for my in -12i16..12 {
            for mx in -12i16..12 {
                let mv = Mv::new(mx, my);
                let (bx, by) = (16i32, 8i32);
                let mut buf = [0u8; 256];
                let (p, ps) = me.pred(bx, by, mv, 16, 16, &mut buf);
                let mut want = [0u8; 16 * PRED_STRIDE];
                let xi = bx + (mx as i32 >> 2);
                let yi = by + (my as i32 >> 2);
                let k = dsp.qpel[((my & 3) * 4 + (mx & 3)) as usize];
                let o = r.offset((xi - 2) as isize, (yi - 2) as isize);
                k(&mut want, &r.data[o..], r.stride, 16, 16, 255);
                for y in 0..16 {
                    for x in 0..16 {
                        assert_eq!(
                            p[y * ps + x],
                            want[y * PRED_STRIDE + x],
                            "mv ({mx}, {my}) at ({x}, {y})"
                        );
                    }
                }
            }
        }
    }
}
