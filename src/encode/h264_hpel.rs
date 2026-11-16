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
/// geometry (stride, border) as the plane itself: [`Hpel::plane`] `0`
/// holds the sample half a step right of each integer position (`b`), `1`
/// half a step below (`h` in the standard's naming) and `2` both (`j`).
///
/// The three live in one allocation, each starting at its own offset
/// within a 16 KiB span of addresses ([`HPEL_SKEW`]). Planes allocated
/// apart are each page-aligned, so the same sample of every plane — the
/// integer plane's, the three half planes', every reference's — falls in
/// one set of a 16 KiB-way cache, and a refinement ring reading four
/// planes of three references at one position asks a set for more lines
/// than it has ways. Skewed, they spread over four sets.
pub(crate) struct Hpel<S: Sample> {
    buf: Vec<S>,
    base: [usize; 3],
    len: usize,
}

/// The cache span the half planes are spread over, in bytes: the L1 data
/// cache's way size on the CPUs this was measured on (128 KiB, 8 ways),
/// and a multiple of every smaller one's.
const HPEL_SPAN: usize = 16 << 10;

/// Each half plane's start, as a fraction of [`HPEL_SPAN`] past the
/// integer plane's (which an allocator hands out page-aligned): a quarter,
/// a half and three quarters.
const HPEL_SKEW: [usize; 3] = [HPEL_SPAN / 4, HPEL_SPAN / 2, 3 * HPEL_SPAN / 4];

impl<S: Sample> Hpel<S> {
    /// Half plane `k` (0: `b`, 1: `h`, 2: `j`), laid out exactly as the
    /// integer plane is.
    #[inline(always)]
    pub(crate) fn plane(&self, k: usize) -> &[S] {
        &self.buf[self.base[k]..self.base[k] + self.len]
    }

    /// Three planes of `len` samples, plane `k`'s first sample [`HPEL_SKEW`]`[k]`
    /// bytes past a multiple of [`HPEL_SPAN`].
    fn alloc(len: usize) -> Self {
        let span = HPEL_SPAN / S::BYTES;
        let buf = vec![S::default(); 3 * (len + span)];
        let addr = buf.as_ptr() as usize;
        let mut base = [0usize; 3];
        let mut at = 0usize;
        for k in 0..3 {
            // The first index at or past `at` whose address sits at the
            // skew within the span.
            let want = HPEL_SKEW[k] / S::BYTES;
            let phase = ((addr / S::BYTES) + at) % span;
            at += (want + span - phase) % span;
            base[k] = at;
            at += len;
        }
        Hpel { buf, base, len }
    }
}

#[inline(always)]
fn tap6(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32) -> i32 {
    a + f - 5 * (b + e) + 20 * (c + d)
}

/// Build the half-sample planes of `r`, whose border is already
/// replicated. Positions whose six-tap window would leave the padded plane
/// copy the integer sample instead; the search never reads them (its
/// window keeps every read inside, see [`search_bounds`]). `kernel` is the
/// encode table's 8-bit form of [`hpel_planes`]
/// ([`crate::dsp::h264_enc::H264EncDsp::hpel_planes`]), taken for 8-bit
/// samples where it is set.
pub(crate) fn build_hpel<S: Sample>(
    r: &Recon<S>,
    max: i32,
    kernel: Option<crate::dsp::h264_enc::HpelPlanesFn>,
) -> Hpel<S> {
    let stride = r.stride;
    let len = r.data.len();
    let mut out = Hpel::<S>::alloc(len);
    let [b0, b1, b2] = out.base;
    let (lo, rest) = out.buf.split_at_mut(b1);
    let (mid, hi) = rest.split_at_mut(b2 - b1);
    let h = &mut lo[b0..b0 + len];
    let v = &mut mid[..len];
    let c = &mut hi[..len];
    if let Some(k) = kernel
        && std::any::TypeId::of::<S>() == std::any::TypeId::of::<u8>()
    {
        debug_assert_eq!(max, 255);
        // SAFETY: `S` is `u8`, so these are the same slices.
        let as8 = |p: &mut [S]| unsafe {
            std::slice::from_raw_parts_mut(p.as_mut_ptr() as *mut u8, p.len())
        };
        let d = unsafe { std::slice::from_raw_parts(r.data.as_ptr() as *const u8, len) };
        k(d, stride, as8(h), as8(v), as8(c));
    } else {
        hpel_planes(&r.data, stride, max, h, v, c);
    }
    out
}

/// The three half-sample planes of the padded plane `d` (`stride` wide)
/// into `h`, `v` and `c`, each `d`'s length: the definition the encode
/// table's `hpel_planes` kernels are held to. Positions whose six-tap
/// window would leave the plane hold the integer sample.
pub(crate) fn hpel_planes<S: Sample>(
    d: &[S],
    stride: usize,
    max: i32,
    h: &mut [S],
    v: &mut [S],
    c: &mut [S],
) {
    let rows = d.len() / stride;
    h.copy_from_slice(d);
    v.copy_from_slice(d);
    c.copy_from_slice(d);
    // Horizontal half samples, every row.
    for y in 0..rows {
        let row = &d[y * stride..(y + 1) * stride];
        let out = &mut h[y * stride..(y + 1) * stride];
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
        let vo = &mut v[y * stride..(y + 1) * stride];
        for x in 0..stride {
            vo[x] = S::from_i32(((inter[x] + 16) >> 5).clamp(0, max));
        }
        let co = &mut c[y * stride..(y + 1) * stride];
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
}

/// A reference as the search reads it: the integer plane and its three
/// half-sample planes.
pub(crate) struct MeRef<'a, S: Sample> {
    /// The integer plane, whose geometry every plane shares.
    pub(crate) g: &'a Recon<S>,
    /// `[G, b, h, j]`'s samples: integer, horizontal half, vertical half,
    /// centre, each indexed by `g`'s offsets.
    planes: [&'a [S]; 4],
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
    /// The reference `g` with its half planes `h` (built from it).
    pub(crate) fn new(g: &'a Recon<S>, h: &'a Hpel<S>) -> Self {
        MeRef {
            g,
            planes: [&g.data, h.plane(0), h.plane(1), h.plane(2)],
        }
    }

    #[inline(always)]
    fn at(&self, plane: u8, x: i32, y: i32) -> &'a [S] {
        &self.planes[plane as usize][self.g.offset(x as isize, y as isize)..]
    }

    /// The two plane views whose rounded average is the prediction of the
    /// block at picture position `(x, y)` for vector `mv` (one view twice
    /// where the position is a plane sample: its own average).
    #[inline]
    pub(crate) fn pair(&self, x: i32, y: i32, mv: Mv) -> (&'a [S], &'a [S]) {
        let (xi, yi) = (x + (mv.x as i32 >> 2), y + (mv.y as i32 >> 2));
        let (p1, dx1, dy1, p2, dx2, dy2) = QPEL[((mv.y & 3) * 4 + (mv.x & 3)) as usize];
        let a = self.at(p1, xi + dx1 as i32, yi + dy1 as i32);
        if p2 == 255 {
            (a, a)
        } else {
            (a, self.at(p2, xi + dx2 as i32, yi + dy2 as i32))
        }
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
        let stride = self.g.stride;
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
        a.len() >= (h - 1) * as_ + W
            && b.len() >= (h - 1) * bs + W
            && out.len() >= (h - 1) * os + W
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

/// A `w` by `h` block copied between two strides: fixed-width rows for
/// the 8-bit widths a partition has, not a `memmove` per row.
#[inline]
pub(crate) fn copy_rows<S: Sample>(
    out: &mut [S],
    os: usize,
    a: &[S],
    as_: usize,
    w: usize,
    h: usize,
) {
    #[inline(always)]
    fn rows<S: Sample, const W: usize>(out: &mut [S], os: usize, a: &[S], as_: usize, h: usize) {
        for y in 0..h {
            let r: &[S; W] = a[y * as_..y * as_ + W].try_into().expect("W samples");
            let o: &mut [S; W] = (&mut out[y * os..y * os + W])
                .try_into()
                .expect("W samples");
            *o = *r;
        }
    }
    match w {
        16 => rows::<S, 16>(out, os, a, as_, h),
        8 => rows::<S, 8>(out, os, a, as_, h),
        4 => rows::<S, 4>(out, os, a, as_, h),
        _ => {
            for y in 0..h {
                out[y * os..y * os + w].copy_from_slice(&a[y * as_..y * as_ + w]);
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
    tab: Box<[u16; MV_TAB]>,
}

/// [`MvCostTable`]'s entries: `-MV_SPAN..=MV_SPAN`.
const MV_TAB: usize = 2 * MV_SPAN as usize + 1;

/// The cost of differing from the predictor by `d` in one component, in
/// an [`MvCostTable`]'s entries (no bounds check: the index is clamped
/// to the array's own length).
#[inline(always)]
fn mv_comp(tab: &[u16; MV_TAB], d: i32) -> u32 {
    tab[(d.clamp(-MV_SPAN, MV_SPAN) + MV_SPAN) as usize] as u32
}

impl MvCostTable {
    pub(crate) fn new(lambda: f32) -> Self {
        let mut tab = Box::new([0u16; MV_TAB]);
        for (t, d) in tab.iter_mut().zip(-MV_SPAN..=MV_SPAN) {
            *t = (lambda * mvd_bits(d) + 0.5).min(65535.0) as u16;
        }
        MvCostTable { tab }
    }

    /// The cost of differing from the predictor by `d` in one component.
    #[inline(always)]
    pub(crate) fn comp(&self, d: i32) -> u32 {
        mv_comp(&self.tab, d)
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

impl MeCfg {
    /// No refinement: [`Search::refine`] scores its start and returns it.
    pub(crate) const SCORE: MeCfg = MeCfg {
        range: 0,
        hpel_iters: 0,
        qpel_iters: 0,
        umh: false,
        qpel_diamond: false,
        dia: false,
    };
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
    /// The encode table's fused average-SATD, where the CPU has one.
    pub(crate) satd_avg: Option<crate::dsp::h264_enc::SatdAvgFn>,
    /// The encode table's four-way SAD, where the CPU has one.
    pub(crate) sad_x4: Option<crate::dsp::h264_enc::SadX4Fn>,
    /// The encode table's four-way average-SATD, where the CPU has one.
    pub(crate) satd_x4: Option<crate::dsp::h264_enc::SatdAvgX4Fn>,
    /// The encode table's four-way plane-sample SATD, where the CPU has one.
    pub(crate) satd_plain_x4: Option<crate::dsp::h264_enc::SatdX4Fn>,
}

/// Whole-sample points a search step scores together, at most sixteen.
struct Pts {
    p: [(i32, i32); 16],
    n: usize,
}

impl Pts {
    fn new() -> Self {
        Pts {
            p: [(0, 0); 16],
            n: 0,
        }
    }

    #[inline]
    fn push(&mut self, f: (i32, i32)) {
        self.p[self.n] = f;
        self.n += 1;
    }
}

/// What a search reads for every candidate, read out of the search's
/// references once: past an opaque call (every kernel is one, through
/// its table) the compiler cannot know the planes and the rate table did
/// not move, and reloads each through two or three dependent loads per
/// candidate otherwise.
#[derive(Clone, Copy)]
struct Hot<'a, S: Sample> {
    /// `[G, b, h, j]`, as [`MeRef`] holds them.
    planes: [&'a [S]; 4],
    stride: isize,
    /// The block's first sample, in every plane.
    org: isize,
    tab: &'a [u16; MV_TAB],
    /// The vector predictor.
    pred: (i32, i32),
}

impl<'a, S: Sample> Hot<'a, S> {
    /// Plane `plane` from whole-sample offset `(dx, dy)` of the block.
    #[inline(always)]
    fn at(&self, plane: u8, dx: i32, dy: i32) -> &'a [S] {
        &self.planes[plane as usize]
            [(self.org + dy as isize * self.stride + dx as isize) as usize..]
    }

    /// [`MeRef::pair`] for the block.
    #[inline(always)]
    fn pair(&self, mv: Mv) -> (&'a [S], &'a [S]) {
        let (xi, yi) = (mv.x as i32 >> 2, mv.y as i32 >> 2);
        let (p1, dx1, dy1, p2, dx2, dy2) = QPEL[((mv.y & 3) * 4 + (mv.x & 3)) as usize];
        let a = self.at(p1, xi + dx1 as i32, yi + dy1 as i32);
        if p2 == 255 {
            (a, a)
        } else {
            (a, self.at(p2, xi + dx2 as i32, yi + dy2 as i32))
        }
    }

    /// The rate of the quarter-sample vector `(x, y)`.
    #[inline(always)]
    fn rate(&self, x: i32, y: i32) -> u32 {
        mv_comp(self.tab, x - self.pred.0) + mv_comp(self.tab, y - self.pred.1)
    }
}

impl<S: Sample> Search<'_, '_, S> {
    #[inline(always)]
    fn hot(&self) -> Hot<'_, S> {
        let g = self.r.g;
        Hot {
            planes: self.r.planes,
            stride: g.stride as isize,
            org: g.offset(self.x as isize, self.y as isize) as isize,
            tab: &self.cost.tab,
            pred: (self.pred.x as i32, self.pred.y as i32),
        }
    }

    #[inline]
    fn sad_full(&self, hot: &Hot<'_, S>, fx: i32, fy: i32) -> u32 {
        (self.dist.sad)(
            self.src,
            self.src_stride,
            hot.at(0, fx, fy),
            hot.stride as usize,
            self.w,
            self.h,
        ) + hot.rate(fx * 4, fy * 4)
    }

    /// [`Search::sad_full`] at every point of `pts`, in order: four at a
    /// time through the four-way kernel where there is one.
    #[inline]
    fn sad_pts(&self, hot: &Hot<'_, S>, pts: &Pts) -> [u32; 16] {
        let mut out = [0u32; 16];
        let n = pts.n;
        if let Some(f) = self.sad_x4.filter(|_| S::BYTES == 1 && n > 1) {
            // SAFETY: a one-byte `Sample` is `u8`.
            let as8 =
                |v: &[S]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()) };
            let at = |k: usize| {
                let (fx, fy) = pts.p[k.min(n - 1)];
                as8(hot.at(0, fx, fy))
            };
            let s8 = as8(self.src);
            let mut k = 0;
            while k < n {
                let d = f(
                    s8,
                    self.src_stride,
                    [at(k), at(k + 1), at(k + 2), at(k + 3)],
                    hot.stride as usize,
                    self.w,
                    self.h,
                );
                for j in 0..4.min(n - k) {
                    let (fx, fy) = pts.p[k + j];
                    out[k + j] = d[j] + hot.rate(fx * 4, fy * 4);
                }
                k += 4;
            }
        } else {
            for k in 0..n {
                out[k] = self.sad_full(hot, pts.p[k].0, pts.p[k].1);
            }
        }
        out
    }

    #[inline]
    fn inside(&self, fx: i32, fy: i32) -> bool {
        let (lx, hx, ly, hy) = self.bounds;
        fx >= lx && fx <= hx && fy >= ly && fy <= hy
    }

    /// SATD of the prediction at `mv` (no rate).
    #[inline]
    fn satd(&self, hot: &Hot<'_, S>, mv: Mv, buf: &mut [S; 256]) -> u32 {
        if let Some(f) = self.satd_avg
            && S::BYTES == 1
        {
            let (_, _, _, p2, _, _) = QPEL[((mv.y & 3) * 4 + (mv.x & 3)) as usize];
            if p2 != 255 {
                let (a, b) = hot.pair(mv);
                let rs = hot.stride as usize;
                // SAFETY: a one-byte `Sample` is `u8`.
                let (s8, a8, b8) = unsafe {
                    (
                        std::slice::from_raw_parts(self.src.as_ptr() as *const u8, self.src.len()),
                        std::slice::from_raw_parts(a.as_ptr() as *const u8, a.len()),
                        std::slice::from_raw_parts(b.as_ptr() as *const u8, b.len()),
                    )
                };
                return f(s8, self.src_stride, a8, rs, b8, rs, self.w, self.h);
            }
        }
        let (p, ps) = self.r.pred(self.x, self.y, mv, self.w, self.h, buf);
        (self.dist.satd)(self.src, self.src_stride, p, ps, self.w, self.h)
    }

    /// The SATDs of the predictions at `mvs[..n]` (no rate), four at a
    /// time through the four-way kernel where there is one — each the
    /// value [`Search::satd`] gives.
    #[inline]
    fn satd_n(
        &self,
        hot: &Hot<'_, S>,
        mvs: &[Mv; 8],
        n: usize,
        plain: bool,
        buf: &mut [S; 256],
    ) -> [u32; 8] {
        let mut out = [0u32; 8];
        // Every point a plane sample (`plain`: a half-sample ring, whose
        // points all have even components): no average to take.
        if let Some(f) = self
            .satd_plain_x4
            .filter(|_| S::BYTES == 1 && n > 1 && plain)
        {
            // SAFETY: a one-byte `Sample` is `u8`.
            let as8 =
                |v: &[S]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()) };
            let at = |k: usize| as8(hot.pair(mvs[k.min(n - 1)]).0);
            let s8 = as8(self.src);
            let mut k = 0;
            while k < n {
                let d = f(
                    s8,
                    self.src_stride,
                    [at(k), at(k + 1), at(k + 2), at(k + 3)],
                    hot.stride as usize,
                    self.w,
                    self.h,
                );
                out[k..(k + 4).min(8)].copy_from_slice(&d[..(8 - k).min(4)]);
                k += 4;
            }
        } else if let Some(f) = self.satd_x4.filter(|_| S::BYTES == 1 && n > 1) {
            let rs = hot.stride as usize;
            // SAFETY: a one-byte `Sample` is `u8`.
            let as8 =
                |v: &[S]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()) };
            // A plane sample is the pair of one view twice: its rounded
            // average is itself.
            let pair = |k: usize| {
                let (a, b) = hot.pair(mvs[k.min(n - 1)]);
                (as8(a), as8(b))
            };
            let s8 = as8(self.src);
            let mut k = 0;
            while k < n {
                let d = f(
                    s8,
                    self.src_stride,
                    [pair(k), pair(k + 1), pair(k + 2), pair(k + 3)],
                    rs,
                    self.w,
                    self.h,
                );
                out[k..(k + 4).min(8)].copy_from_slice(&d[..(8 - k).min(4)]);
                k += 4;
            }
        } else {
            for k in 0..n {
                out[k] = self.satd(hot, mvs[k], buf);
            }
        }
        out
    }

    /// Run the search from `cands` (quarter-sample vectors; rounded to
    /// whole samples for the descent). Returns the vector, its SATD and
    /// its full cost (SATD plus rate).
    pub(crate) fn run(&self, cands: &[Mv], cfg: &MeCfg) -> (Mv, u32, u32) {
        let (mv, _) = self.int_search(cands, cfg);
        self.refine(mv, cfg)
    }

    /// The whole-sample stage of [`Search::run`]: the best whole-sample
    /// vector (in quarter samples) and its cost, SAD plus rate.
    pub(crate) fn int_search(&self, cands: &[Mv], cfg: &MeCfg) -> (Mv, u32) {
        let hot = self.hot();
        let (lx, hx, ly, hy) = self.bounds;
        let clampf = |m: Mv| {
            (
                ((m.x as i32 + 2) >> 2).clamp(lx, hx),
                ((m.y as i32 + 2) >> 2).clamp(ly, hy),
            )
        };
        let mut best = clampf(self.pred);
        // The predictor and the distinct candidates, scored together and
        // compared in order (each point's score is its own, so the order
        // they were computed in changes nothing).
        let mut pts = Pts::new();
        pts.push(best);
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
            // Callers pass at most eight candidates; the batch holds sixteen.
            pts.push(f);
        }
        let sc = self.sad_pts(&hot, &pts);
        let mut bcost = sc[0];
        for k in 1..pts.n {
            if sc[k] < bcost {
                bcost = sc[k];
                best = pts.p[k];
            }
        }
        let start = best;
        let (bx0, bx1) = ((start.0 - cfg.range).max(lx), (start.0 + cfg.range).min(hx));
        let (by0, by1) = ((start.1 - cfg.range).max(ly), (start.1 + cfg.range).min(hy));
        let ok = |fx: i32, fy: i32| fx >= bx0 && fx <= bx1 && fy >= by0 && fy <= by1;
        // Points around a fixed centre `c`, the ones inside the window,
        // scored together and taken in order where strictly cheaper (as
        // one at a time would: no two are the same point, and none is the
        // centre). Returns the index of the last one taken.
        let ring = |c: (i32, i32), offs: &[(i32, i32)], best: &mut (i32, i32), bcost: &mut u32| {
            let mut pts = Pts::new();
            let mut idx = [0usize; 16];
            for (k, &(dx, dy)) in offs.iter().enumerate() {
                let (fx, fy) = (c.0 + dx, c.1 + dy);
                if ok(fx, fy) && (fx, fy) != c {
                    idx[pts.n] = k;
                    pts.push((fx, fy));
                }
            }
            let sc = self.sad_pts(&hot, &pts);
            let mut moved = usize::MAX;
            for k in 0..pts.n {
                if sc[k] < *bcost {
                    *bcost = sc[k];
                    *best = pts.p[k];
                    moved = idx[k];
                }
            }
            moved
        };
        let try_at = |fx: i32, fy: i32, best: &mut (i32, i32), bcost: &mut u32| {
            if ok(fx, fy) && (fx, fy) != *best {
                let s = self.sad_full(&hot, fx, fy);
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
            // The rings' four points on the axes, (+-r, 0) and (0, +-r),
            // are the cross's own (every ring radius is a multiple of four,
            // the cross's of two), already scored against this centre.
            let mut r = 4;
            while r <= cfg.range {
                for (dx, dy) in [
                    (-2, 1),
                    (-2, -1),
                    (2, 1),
                    (2, -1),
                    (-1, 2),
                    (1, 2),
                    (-1, -2),
                    (1, -2),
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
                ring(
                    c,
                    &[(-1, 0), (1, 0), (0, -1), (0, 1)],
                    &mut best,
                    &mut bcost,
                );
                if best == c {
                    break;
                }
            }
            return (Mv::new((best.0 * 4) as i16, (best.1 * 4) as i16), bcost);
        }
        // Hexagon descent. After the first ring, a move to one of the six
        // vertices leaves three of the new ring's points already scored
        // (they were the old ring's), so only the other three are tried.
        const HEX: [(i32, i32); 6] = [(-2, 0), (-1, -2), (1, -2), (2, 0), (1, 2), (-1, 2)];
        let c = best;
        let mut dir = ring(c, &HEX, &mut best, &mut bcost);
        let mut it = 1;
        while dir != usize::MAX && it < cfg.range {
            let c = best;
            let ks = [(dir + 5) % 6, dir, (dir + 1) % 6];
            let moved = ring(c, &ks.map(|k| HEX[k]), &mut best, &mut bcost);
            dir = if moved == usize::MAX {
                moved
            } else {
                ks[moved]
            };
            it += 1;
        }
        // Square refinement.
        let c = best;
        ring(
            c,
            &[
                (-1, -1),
                (0, -1),
                (1, -1),
                (-1, 0),
                (1, 0),
                (-1, 1),
                (0, 1),
                (1, 1),
            ],
            &mut best,
            &mut bcost,
        );
        debug_assert!(self.inside(best.0, best.1));
        (Mv::new((best.0 * 4) as i16, (best.1 * 4) as i16), bcost)
    }

    /// Sub-sample refinement from `mv`: square rings at half, then quarter
    /// samples, scored by SATD plus rate. Returns `(mv, satd, cost)`.
    pub(crate) fn refine(&self, mv: Mv, cfg: &MeCfg) -> (Mv, u32, u32) {
        let hot = self.hot();
        let (lx, hx, ly, hy) = self.bounds;
        let (lx, hx, ly, hy) = (lx * 4, hx * 4, ly * 4, hy * 4);
        let mut buf = [S::default(); 256];
        let mut best = mv;
        let mut bsatd = self.satd(&hot, best, &mut buf);
        let mut bcost = bsatd + hot.rate(best.x as i32, best.y as i32);
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
        // A second ring around a neighbour of the first's centre shares
        // three to five of its points (those within one step of the old
        // centre), and each of those lost to a cost no lower than the best
        // now, so it would lose again: they are skipped, which changes
        // nothing but the time. (The quarter-sample ring shares none with
        // the half-sample ones: each of its points has an odd component.)
        for (step, iters) in [(2i16, cfg.hpel_iters), (1, cfg.qpel_iters)] {
            // Quarter samples: the diamond, when asked.
            let pts = if step == 1 && cfg.qpel_diamond { 4 } else { 8 };
            let mut prev: Option<Mv> = None;
            for _ in 0..iters {
                let c = best;
                // The ring's points are scored together (four at a time),
                // then taken in order where strictly cheaper, as one at a
                // time would: a point's score is its own. (A point whose
                // rate alone reaches the best cost cannot win; scoring it
                // anyway changes nothing.)
                let mut ms = [Mv::ZERO; 8];
                let mut n = 0;
                for &(dx, dy) in &RING[..pts] {
                    let m = Mv::new(c.x + dx * step, c.y + dy * step);
                    let (x, y) = (m.x as i32, m.y as i32);
                    if !(x >= lx && x <= hx && y >= ly && y <= hy)
                        || prev
                            .is_some_and(|p| (m.x - p.x).abs() <= step && (m.y - p.y).abs() <= step)
                    {
                        continue;
                    }
                    ms[n] = m;
                    n += 1;
                }
                let ss = self.satd_n(&hot, &ms, n, step == 2, &mut buf);
                for (&m, &s) in ms[..n].iter().zip(&ss[..n]) {
                    let rc = hot.rate(m.x as i32, m.y as i32);
                    if s + rc < bcost {
                        bcost = s + rc;
                        bsatd = s;
                        best = m;
                    }
                }
                if best == c {
                    break;
                }
                prev = Some(c);
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
        let hp = build_hpel(
            &r,
            255,
            crate::dsp::h264_enc::H264EncDsp::new(Cpu::detect()).hpel_planes,
        );
        let me = MeRef::new(&r, &hp);
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
