//! A lookahead for the H.264 encoder: half-resolution cost estimates of
//! the pictures not yet coded, and from them a quantiser offset for every
//! macroblock of a reference picture according to how much of it the
//! pictures after it will reuse.
//!
//! The idea is the reference-propagation one (a macroblock that many later
//! macroblocks predict from is worth coding better, because its error is
//! inherited by all of them; one nothing predicts from is not), built here
//! from first principles:
//!
//! - Each picture is reduced to half resolution, and for each 8x8 block of
//!   it — one macroblock of the picture — the lookahead estimates an intra
//!   cost (SATD of the best of four simple predictions from source
//!   neighbours) and, against the reference(s) it will be coded from, an
//!   inter cost and vector (a small integer search, SATD).
//! - Walking from the last picture in the window back to the one about to
//!   be coded, a block passes to the reference blocks its vector lands on
//!   the share of its information it does not have to code itself:
//!   `(intra + inherited) * (1 - inter / intra)`, split over the up-to-four
//!   blocks the displaced block overlaps by area (and halved per list for
//!   a bi-predicted one).
//! - A macroblock that so inherits `p` against its own intra cost `c` is
//!   coded at `strength * log2(1 + p / c)` below the picture quantiser.
//!
//! Only reference pictures get offsets: a non-reference B picture has no
//! descendants.

use crate::dsp::distortion::DistortionDsp;
use crate::encode::gop::{Coded, Kind};
use crate::encode::h264_syntax::Recon;
use crate::sample::Sample;
use std::collections::{BTreeMap, HashMap};

/// Border of the half-resolution planes, for the search.
const PAD: usize = 24;
/// Integer search range at half resolution.
const RANGE: i32 = 16;

/// One picture at half resolution, with its intra costs.
struct LaPic<S: Sample> {
    low: Recon<S>,
    intra: Vec<u32>,
}

/// A block's estimate against its references: cost, and per list the
/// vector at half resolution where that list is used.
#[derive(Clone, Copy, Default)]
struct BlockCost {
    cost: u32,
    mv: [(i16, i16); 2],
    used: [bool; 2],
}

/// The lookahead's state: pictures by display index, and the estimates
/// made so far (they do not change, so each is made once).
pub(crate) struct Lookahead<S: Sample> {
    pub(crate) depth: usize,
    strength: f32,
    mbs_wide: usize,
    mbs_high: usize,
    pics: BTreeMap<u64, LaPic<S>>,
    /// `(picture, ref0, ref1)`: one list (`ref1 == u64::MAX`) or two.
    costs: HashMap<(u64, u64, u64), Vec<BlockCost>>,
    dist: DistortionDsp<S>,
}

impl<S: Sample> Lookahead<S> {
    pub(crate) fn new(depth: usize, strength: f32, mbs_wide: usize, mbs_high: usize) -> Self {
        Lookahead {
            depth,
            strength,
            mbs_wide,
            mbs_high,
            pics: BTreeMap::new(),
            costs: HashMap::new(),
            dist: DistortionDsp::new(crate::dsp::Cpu::detect_honouring_env()),
        }
    }

    /// Take picture `display`'s luma (coded size, `stride` wide, edge
    /// replicated to the macroblock grid by the caller's padding or here).
    pub(crate) fn add(&mut self, display: u64, luma: &[S], width: usize, height: usize) {
        let (lw, lh) = (self.mbs_wide * 8, self.mbs_high * 8);
        let mut low = Recon::<S>::new(lw, lh, PAD);
        for y in 0..lh {
            let y0 = (2 * y).min(height - 1);
            let y1 = (2 * y + 1).min(height - 1);
            for x in 0..lw {
                let x0 = (2 * x).min(width - 1);
                let x1 = (2 * x + 1).min(width - 1);
                let s = luma[y0 * width + x0].to_i32()
                    + luma[y0 * width + x1].to_i32()
                    + luma[y1 * width + x0].to_i32()
                    + luma[y1 * width + x1].to_i32();
                let o = low.offset(x as isize, y as isize);
                low.data[o] = S::from_i32((s + 2) >> 2);
            }
        }
        low.extend_edges(false);
        let intra = self.intra_costs(&low);
        self.pics.insert(display, LaPic { low, intra });
    }

    /// Forget every picture before `display` (and the estimates naming
    /// them).
    pub(crate) fn drop_before(&mut self, display: u64) {
        self.pics.retain(|&d, _| d >= display);
        self.costs.retain(|&(a, b, c), _| {
            a >= display && b >= display && (c == u64::MAX || c >= display)
        });
    }

    fn intra_costs(&self, low: &Recon<S>) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.mbs_wide * self.mbs_high);
        let mut pred = [S::default(); 64];
        for by in 0..self.mbs_high {
            for bx in 0..self.mbs_wide {
                let (x, y) = ((bx * 8) as isize, (by * 8) as isize);
                let o = low.offset(x, y);
                let src = &low.data[o..];
                let st = low.stride;
                let mut best = u32::MAX;
                let top: Option<[i32; 8]> = (by > 0).then(|| {
                    let mut t = [0; 8];
                    for (i, v) in t.iter_mut().enumerate() {
                        *v = low.data[low.offset(x + i as isize, y - 1)].to_i32();
                    }
                    t
                });
                let left: Option<[i32; 8]> = (bx > 0).then(|| {
                    let mut l = [0; 8];
                    for (i, v) in l.iter_mut().enumerate() {
                        *v = low.data[low.offset(x - 1, y + i as isize)].to_i32();
                    }
                    l
                });
                // DC (from whatever neighbours exist, mid-grey with none).
                let dc = match (top, left) {
                    (Some(t), Some(l)) => (t.iter().sum::<i32>() + l.iter().sum::<i32>() + 8) >> 4,
                    (Some(t), None) => (t.iter().sum::<i32>() + 4) >> 3,
                    (None, Some(l)) => (l.iter().sum::<i32>() + 4) >> 3,
                    (None, None) => {
                        // No neighbours: the block's own mean stands in.
                        let mut s = 0;
                        for yy in 0..8 {
                            for xx in 0..8 {
                                s += src[yy * st + xx].to_i32();
                            }
                        }
                        (s + 32) >> 6
                    }
                };
                pred.fill(S::from_i32(dc));
                best = best.min((self.dist.satd)(src, st, &pred, 8, 8, 8));
                if let Some(t) = top {
                    for yy in 0..8 {
                        for xx in 0..8 {
                            pred[yy * 8 + xx] = S::from_i32(t[xx]);
                        }
                    }
                    best = best.min((self.dist.satd)(src, st, &pred, 8, 8, 8));
                }
                if let Some(l) = left {
                    for yy in 0..8 {
                        for xx in 0..8 {
                            pred[yy * 8 + xx] = S::from_i32(l[yy]);
                        }
                    }
                    best = best.min((self.dist.satd)(src, st, &pred, 8, 8, 8));
                }
                if let (Some(t), Some(l)) = (top, left) {
                    // A bilinear blend of the two edges.
                    for yy in 0..8 {
                        for xx in 0..8 {
                            let v = (t[xx] * (8 - yy as i32)
                                + l[yy] * (8 - xx as i32)
                                + t[7] * yy as i32
                                + l[7] * xx as i32
                                + 8)
                                >> 4;
                            pred[yy * 8 + xx] = S::from_i32(v);
                        }
                    }
                    best = best.min((self.dist.satd)(src, st, &pred, 8, 8, 8));
                }
                out.push(best + 12);
            }
        }
        out
    }

    /// Integer search of block `(bx, by)` of `cur` in `refp`, from the
    /// candidates: `(cost, mv)` at SAD, the winner rescored by SATD.
    fn search(
        &self,
        cur: &Recon<S>,
        refp: &Recon<S>,
        bx: usize,
        by: usize,
        cands: &[(i16, i16)],
    ) -> (u32, (i16, i16)) {
        let (x, y) = ((bx * 8) as i32, (by * 8) as i32);
        let src = &cur.data[cur.offset(x as isize, y as isize)..];
        let lim = (PAD as i32) - 8;
        let (lx, hx) = (
            (-x - lim).max(-RANGE * 2),
            (cur.width as i32 - 8 - x + lim).min(RANGE * 2),
        );
        let (ly, hy) = (
            (-y - lim).max(-RANGE * 2),
            (cur.height as i32 - 8 - y + lim).min(RANGE * 2),
        );
        let sad = |mx: i32, my: i32| -> u32 {
            let o = refp.offset((x + mx) as isize, (y + my) as isize);
            (self.dist.sad)(src, cur.stride, &refp.data[o..], refp.stride, 8, 8)
                + 2 * (mx.unsigned_abs() + my.unsigned_abs()).min(64)
        };
        let mut best = (sad(0, 0), (0, 0));
        for &(cx, cy) in cands {
            let (cx, cy) = ((cx as i32).clamp(lx, hx), (cy as i32).clamp(ly, hy));
            if (cx, cy) != best.1 {
                let s = sad(cx, cy);
                if s < best.0 {
                    best = (s, (cx, cy));
                }
            }
        }
        for _ in 0..RANGE {
            let c = best.1;
            for (dx, dy) in [(-2, 0), (-1, -2), (1, -2), (2, 0), (1, 2), (-1, 2)] {
                let (mx, my) = (c.0 + dx, c.1 + dy);
                if mx < lx || mx > hx || my < ly || my > hy {
                    continue;
                }
                let s = sad(mx, my);
                if s < best.0 {
                    best = (s, (mx, my));
                }
            }
            if best.1 == c {
                break;
            }
        }
        let c = best.1;
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
            let (mx, my) = (c.0 + dx, c.1 + dy);
            if mx < lx || mx > hx || my < ly || my > hy {
                continue;
            }
            let s = sad(mx, my);
            if s < best.0 {
                best = (s, (mx, my));
            }
        }
        let (mx, my) = best.1;
        let o = refp.offset((x + mx) as isize, (y + my) as isize);
        let satd = (self.dist.satd)(src, cur.stride, &refp.data[o..], refp.stride, 8, 8);
        (satd, (mx as i16, my as i16))
    }

    /// The estimates of picture `d` against `r0` alone, or the pair
    /// `(r0, r1)` (best of either list or their average per block).
    fn estimate(&mut self, d: u64, r0: u64, r1: Option<u64>) -> bool {
        let key = (d, r0, r1.unwrap_or(u64::MAX));
        if self.costs.contains_key(&key) {
            return true;
        }
        let (Some(cur), Some(p0)) = (self.pics.get(&d), self.pics.get(&r0)) else {
            return false;
        };
        let p1 = match r1 {
            Some(r) => match self.pics.get(&r) {
                Some(p) => Some(p),
                None => return false,
            },
            None => None,
        };
        let (w, h) = (self.mbs_wide, self.mbs_high);
        let mut out = vec![BlockCost::default(); w * h];
        let mut avg = [S::default(); 64];
        for by in 0..h {
            for bx in 0..w {
                let i = by * w + bx;
                let mut cands0 = [(0i16, 0i16); 2];
                let mut cands1 = [(0i16, 0i16); 2];
                if bx > 0 {
                    cands0[0] = out[i - 1].mv[0];
                    cands1[0] = out[i - 1].mv[1];
                }
                if by > 0 {
                    cands0[1] = out[i - w].mv[0];
                    cands1[1] = out[i - w].mv[1];
                }
                let (c0, m0) = self.search(&cur.low, &p0.low, bx, by, &cands0);
                let mut bc = BlockCost {
                    cost: c0,
                    mv: [m0, (0, 0)],
                    used: [true, false],
                };
                if let Some(p1) = p1 {
                    let (c1, m1) = self.search(&cur.low, &p1.low, bx, by, &cands1);
                    let (x, y) = ((bx * 8) as isize, (by * 8) as isize);
                    let a = &p0.low.data[p0.low.offset(x + m0.0 as isize, y + m0.1 as isize)..];
                    let b = &p1.low.data[p1.low.offset(x + m1.0 as isize, y + m1.1 as isize)..];
                    crate::encode::h264_hpel::avg_into(
                        &mut avg,
                        8,
                        a,
                        p0.low.stride,
                        b,
                        p1.low.stride,
                        8,
                        8,
                    );
                    let src = &cur.low.data[cur.low.offset(x, y)..];
                    let cb = (self.dist.satd)(src, cur.low.stride, &avg, 8, 8, 8);
                    bc.mv[1] = m1;
                    if c1 < bc.cost {
                        bc = BlockCost {
                            cost: c1,
                            mv: [m0, m1],
                            used: [false, true],
                        };
                    }
                    if cb < bc.cost {
                        bc = BlockCost {
                            cost: cb,
                            mv: [m0, m1],
                            used: [true, true],
                        };
                    }
                }
                out[i] = bc;
            }
        }
        self.costs.insert(key, out);
        true
    }

    /// Spread `amount[i]` of every block of a picture whose estimate is
    /// `key` into `into` (the list-`l` reference's accumulator), through
    /// each block's vector, by overlap area.
    fn spread(&self, est: &[BlockCost], amount: &[f32], l: usize, into: &mut [f32]) {
        let (w, h) = (self.mbs_wide as i32, self.mbs_high as i32);
        for (i, bc) in est.iter().enumerate() {
            if !bc.used[l] || amount[i] <= 0.0 {
                continue;
            }
            let share = if bc.used[0] && bc.used[1] { 0.5 } else { 1.0 };
            let a = amount[i] * share;
            let (bx, by) = ((i as i32 % w) * 8, (i as i32 / w) * 8);
            let (x, y) = (bx + bc.mv[l].0 as i32, by + bc.mv[l].1 as i32);
            let (x0, y0) = (x.div_euclid(8), y.div_euclid(8));
            let (fx, fy) = (x.rem_euclid(8), y.rem_euclid(8));
            for (dx, dy, area) in [
                (0, 0, (8 - fx) * (8 - fy)),
                (1, 0, fx * (8 - fy)),
                (0, 1, (8 - fx) * fy),
                (1, 1, fx * fy),
            ] {
                let (tx, ty) = (x0 + dx, y0 + dy);
                if area == 0 || tx < 0 || ty < 0 || tx >= w || ty >= h {
                    continue;
                }
                into[(ty * w + tx) as usize] += a * area as f32 / 64.0;
            }
        }
    }

    /// The quantiser offsets of reference picture `cur`, given the display
    /// indices of the references coded before it (`before`, which the
    /// pictures after it may also predict from) and the pictures coded
    /// after it that the scheduler has placed (`later`, coding order).
    /// Each later picture predicts from the nearest reference coded before
    /// it on each side it may use — the encoder's own list choice — and
    /// the walk stops at the window's edge or at an intra picture, past
    /// which nothing predicts across. `None` when nothing propagates.
    pub(crate) fn offsets(
        &mut self,
        cur: Coded,
        before: &[u64],
        later: &[Coded],
    ) -> Option<Vec<i32>> {
        if !self.pics.contains_key(&cur.display) {
            return None;
        }
        let n = self.mbs_wide * self.mbs_high;
        let mut coded: Vec<u64> = before.to_vec();
        coded.push(cur.display);
        // (display, is a reference, list-0 ref, list-1 ref)
        let mut plan: Vec<(u64, bool, u64, Option<u64>)> = Vec::new();
        for c in later {
            if c.kind.is_intra() || !self.pics.contains_key(&c.display) {
                break;
            }
            let l0 = coded.iter().copied().filter(|&d| d < c.display).max();
            let l1 = coded.iter().copied().filter(|&d| d > c.display).min();
            let Some(l0) = l0 else { break };
            let l1 = if c.kind == Kind::B { l1 } else { None };
            if c.kind == Kind::B && l1.is_none() {
                break;
            }
            plan.push((c.display, c.reference, l0, l1));
            if c.reference {
                coded.push(c.display);
            }
        }
        // The accumulators: the current picture and every later reference.
        let mut prop: HashMap<u64, Vec<f32>> = HashMap::new();
        prop.insert(cur.display, vec![0.0; n]);
        for p in &plan {
            if p.1 {
                prop.insert(p.0, vec![0.0; n]);
            }
        }
        let mut amount = vec![0f32; n];
        for &(d, reference, l0, l1) in plan.iter().rev() {
            if !self.estimate(d, l0, l1) {
                continue;
            }
            let key = (d, l0, l1.unwrap_or(u64::MAX));
            let est = &self.costs[&key];
            let intra = &self.pics[&d].intra;
            let inherited = if reference { prop.get(&d) } else { None };
            for i in 0..n {
                let c = est[i].cost.min(intra[i]) as f32;
                let own = intra[i] as f32 + inherited.map_or(0.0, |v| v[i]);
                amount[i] = own * (1.0 - c / intra[i] as f32);
            }
            for (l, r) in [(0usize, Some(l0)), (1, l1)] {
                if let Some(r) = r {
                    if let Some(mut acc) = prop.remove(&r) {
                        self.spread(est, &amount, l, &mut acc);
                        prop.insert(r, acc);
                    }
                }
            }
        }
        let acc = &prop[&cur.display];
        let intra = &self.pics[&cur.display].intra;
        let out: Vec<i32> = (0..n)
            .map(|i| {
                let r = (acc[i] / intra[i] as f32).max(0.0);
                -(self.strength * (1.0 + r).log2()).round() as i32
            })
            .collect();
        out.iter().any(|&o| o != 0).then_some(out)
    }
}
