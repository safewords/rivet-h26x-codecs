//! Rate-distortion macroblock analysis for progressive H.264 pictures.
//!
//! The decisions the picture walks of [`super::h264_pic`] make for a
//! progressive frame picture, in two stages, the way a production encoder
//! has to make them to be both good and affordable:
//!
//! 1. **Estimate.** Motion is searched on the half-sample planes of
//!    [`super::h264_hpel`] at SATD plus `lambda_satd` times the vectors'
//!    estimated rate; partitions, B directions and intra modes are ranked
//!    on the same footing, with the syntax they add priced in estimated
//!    bits.
//! 2. **Decide.** The few candidates that survive — skip, the best inter
//!    shapes at both transform sizes, the best intra type — are coded for
//!    real, reconstructed, and compared on `SSD + lambda_ssd * R`, where
//!    `R` is what the picture's own entropy writer reports the macroblock
//!    would cost *in its current state* (the walk's `emit` in trial mode:
//!    a counting CABAC encoder over a copy of the live contexts, or a
//!    counting CAVLC writer). No table of what syntax "usually" costs is
//!    involved in the final choice.
//!
//! Everything a decision produces is the same data the older decision
//! functions produced ([`InterDecision`], [`BDecision`], [`MbDecision`]),
//! committed and serialised by the same walks and writers, so the
//! bitstream machinery — and the SELF / CROSS gates over it — is shared.

use crate::encode::h264_hpel::{MeCfg, MeRef, MvCostTable, Search, avg_into, search_bounds};
use crate::encode::h264_intra::{
    IntraCtx, MbAvail, MbDecision, MbKind, as_syntax, avail_4x4, avail_8x8, code_block_4x4,
    code_block_8x8, code_chroma, code_i4x4_plane_fixed, code_i8x8_plane_fixed, code_i16x16,
    code_macroblock_modes8, gather, predicted_mode, quad_rasters, reconstruct_4x4, reconstruct_8x8,
};
use crate::encode::h264_me::{
    BDecision, BMbKind, BRefs, InterDecision, InterMbKind, MbMotionState, PRef,
    code_inter_mb_residual, code_inter_mb_residual_parts, predict_inter_rect, rect_satd_under,
    spatial_direct,
};
use crate::encode::h264_pic::{BMb, Colocated, PMb};
use crate::encode::h264_syntax::Recon;
use crate::encode::h264_trellis::Trellis;
use crate::h264::cavlc::mb_partitions;
use crate::h264::frame::Mv;
use crate::h264::inter::Weighting;
use crate::h264::intra::{predict_4x4, predict_8x8, predict_16x16, predict_chroma};
use crate::h264::mb::{PRED_BI, PRED_L0, PRED_L1, SubMbShape};
use crate::sample::Sample;

/// How much analysis a macroblock gets — one per preset tier.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AnalysisCfg {
    /// The motion search.
    pub(crate) me: MeCfg,
    /// The search for partitions smaller than 16x16 (range and rounds).
    pub(crate) me_sub: MeCfg,
    /// Try `P_Skip` / `B_Skip` before any search, and take it when the
    /// skip prediction leaves no residual.
    pub(crate) early_skip: bool,
    /// ... and when its luma squared error is under this many squared
    /// quantiser steps per sample: a prediction that quantises to nothing
    /// can still be one a searched vector beats by far.
    pub(crate) skip_ssd: f32,
    /// Intra_4x4 is analysed.
    pub(crate) i4x4: bool,
    /// How far the best intra estimate may trail the best inter one and
    /// still be coded for the rate-distortion comparison.
    pub(crate) intra_ratio: f32,
    /// Second-best inter shapes within this ratio of the best estimate are
    /// also coded for the comparison.
    pub(crate) rd_ratio: f32,
    /// At most this many inter shapes are coded for the comparison.
    pub(crate) rd_max: usize,
    /// Re-quantise the chosen macroblock by the trellis (CABAC streams).
    pub(crate) trellis: bool,
    /// Quantise every candidate the comparison codes by the trellis too,
    /// so the decision sees the levels it would write (CABAC streams).
    pub(crate) trellis_rd: bool,
    /// Decide by coding candidates and pricing them (`SSD + lambda * R`);
    /// false decides on the estimates alone and codes only the winner.
    pub(crate) rd: bool,
    /// Active list-0 references of a P picture, at most.
    pub(crate) refs: usize,
    /// A P partition below 16x16 searches every reference whose 16x16
    /// cost came within this factor of the best one's (1.0: the best
    /// 16x16 reference only).
    pub(crate) mixed_refs: f32,
    /// Rounds of joint quarter-sample refinement of a B pair.
    pub(crate) bi_refine: u8,
    /// A partition's pair is refined only when it starts within this
    /// factor of the partition's better single list (0: never).
    pub(crate) bi_part_ratio: f32,
}

impl AnalysisCfg {
    /// The tier closest to x264's `medium`.
    pub(crate) const MEDIUM: AnalysisCfg = AnalysisCfg {
        me: MeCfg {
            range: 16,
            hpel_iters: 2,
            qpel_iters: 2,
            umh: false,
            qpel_diamond: false,
            dia: false,
        },
        me_sub: MeCfg {
            range: 4,
            hpel_iters: 1,
            qpel_iters: 1,
            umh: false,
            qpel_diamond: false,
            dia: false,
        },
        early_skip: true,
        skip_ssd: 0.02,
        i4x4: true,
        intra_ratio: 1.2,
        rd_ratio: 1.1,
        rd_max: 2,
        trellis: true,
        trellis_rd: false,
        rd: true,
        refs: 3,
        mixed_refs: 1.25,
        bi_refine: 2,
        bi_part_ratio: 0.0,
    };

    /// The fastest tier: a diamond search, one refinement round each, one
    /// inter shape priced against skip and intra. (Deciding on the
    /// estimates alone, coding only the winner, measured +35 % BD-rate
    /// against x264 veryfast; pricing one shape, -1.8 %.)
    pub(crate) const VERYFAST: AnalysisCfg = AnalysisCfg {
        me: MeCfg {
            range: 16,
            hpel_iters: 1,
            qpel_iters: 1,
            umh: false,
            qpel_diamond: true,
            dia: true,
        },
        me_sub: MeCfg {
            range: 2,
            hpel_iters: 1,
            qpel_iters: 1,
            umh: false,
            qpel_diamond: true,
            dia: true,
        },
        early_skip: true,
        skip_ssd: 0.06,
        i4x4: true,
        intra_ratio: 1.0,
        rd_ratio: 1.0,
        rd_max: 1,
        trellis: true,
        trellis_rd: false,
        rd: true,
        refs: 1,
        mixed_refs: 1.0,
        bi_refine: 1,
        bi_part_ratio: 0.0,
    };

    /// The most thorough tier: the multi-hexagon search, more refinement,
    /// more candidates priced.
    pub(crate) const SLOW: AnalysisCfg = AnalysisCfg {
        me: MeCfg {
            range: 24,
            hpel_iters: 2,
            qpel_iters: 2,
            umh: true,
            qpel_diamond: false,
            dia: false,
        },
        me_sub: MeCfg {
            range: 8,
            hpel_iters: 2,
            qpel_iters: 2,
            umh: false,
            qpel_diamond: false,
            dia: false,
        },
        early_skip: true,
        skip_ssd: 0.01,
        i4x4: true,
        intra_ratio: 1.5,
        rd_ratio: 1.25,
        rd_max: 3,
        trellis: true,
        trellis_rd: true,
        rd: true,
        refs: 5,
        mixed_refs: 1.5,
        bi_refine: 2,
        bi_part_ratio: 1.2,
    };

    /// The tier for a preset.
    pub(crate) fn of(p: crate::encode::Preset) -> AnalysisCfg {
        match p {
            crate::encode::Preset::Veryfast => Self::VERYFAST,
            crate::encode::Preset::Medium => Self::MEDIUM,
            crate::encode::Preset::Slow => Self::SLOW,
        }
    }
}

/// The squared-error multiplier at `qp`: `0.68 * 2^((qp - 12) / 3)` — the
/// customary `0.85` scaled by 0.8, which measured better against 1.0,
/// 0.65 and 1.25 on the parity corpus (+6.41 % against +7.19 / +6.80 /
/// +8.45 % BD-rate).
pub(crate) fn lambda_ssd(qp: i32) -> f32 {
    0.68 * ((qp - 12) as f32 / 3.0).exp2()
}

/// The multiplier for SATD-scale costs: the square root of the customary
/// `0.85 * 2^((qp - 12) / 3)`, since SATD grows as the error, not its
/// square. (Scaling it with [`lambda_ssd`]'s 0.8, or by 0.8 or 1.25 on
/// its own, measured no better.)
pub(crate) fn lambda_satd(qp: i32) -> f32 {
    (0.85 * ((qp - 12) as f32 / 3.0).exp2()).sqrt()
}

/// What one picture's analysis keeps: the settings, each list's search
/// reference, and the vector-rate tables per quantiser.
pub(crate) struct PicAn<'a, S: Sample> {
    pub(crate) cfg: AnalysisCfg,
    pub(crate) me: [Option<MeRef<'a, S>>; 2],
    /// A P picture's further list-0 references, by index from 1.
    pub(crate) me_more: Vec<MeRef<'a, S>>,
    costs: Vec<Option<MvCostTable>>,
    /// The trellis's context states and multiplier, when it runs.
    trellis: Option<Trellis>,
}

impl<'a, S: Sample> PicAn<'a, S> {
    pub(crate) fn new(cfg: AnalysisCfg, me: [Option<MeRef<'a, S>>; 2]) -> Self {
        PicAn {
            cfg,
            me,
            me_more: Vec::new(),
            costs: (0..52).map(|_| None).collect(),
            trellis: None,
        }
    }

    /// Run the trellis from these CABAC context states (a CABAC picture's
    /// at its start), when the configuration asks for it.
    pub(crate) fn with_trellis(mut self, ctx: Option<[u8; 1024]>, bit_depth: u32) -> Self {
        if self.cfg.trellis {
            self.trellis = ctx.map(|ctx| Trellis {
                ctx,
                lambda: 0.0,
                bd_off: 6 * (bit_depth as i32 - 8),
            });
        }
        self
    }

    /// The trellis's context states, from the writer's as they stand.
    pub(crate) fn refresh_trellis(&mut self, ctx: &[u8; 1024]) {
        if let Some(t) = self.trellis.as_mut() {
            t.ctx.copy_from_slice(ctx);
        }
    }

    /// The trellis, set to the quantiser [`PicAn::prepare`] last saw.
    pub(crate) fn trellis(&self) -> Option<&Trellis> {
        self.trellis.as_ref()
    }

    fn prepare(&mut self, qp: i32) {
        if let Some(t) = self.trellis.as_mut() {
            t.lambda = lambda_ssd(qp);
        }
        let q = qp.clamp(0, 51) as usize;
        if self.costs[q].is_none() {
            self.costs[q] = Some(MvCostTable::new(lambda_satd(qp)));
        }
    }

    fn cost(&self, qp: i32) -> &MvCostTable {
        self.costs[qp.clamp(0, 51) as usize]
            .as_ref()
            .expect("prepared")
    }
}

/// The source planes, as the walks hold them.
pub(crate) struct SrcMb<'s, S: Sample> {
    pub(crate) y: &'s [S],
    pub(crate) ys: usize,
    pub(crate) c: [&'s [S]; 2],
    pub(crate) cs: usize,
}

/// What a macroblock's neighbourhood offers: intra availability and
/// modes, and candidate vectors per list from the coded neighbours.
pub(crate) struct Nb<'n> {
    pub(crate) avail: MbAvail,
    pub(crate) left_modes: &'n [Option<u8>; 4],
    pub(crate) top_modes: &'n [Option<u8>; 4],
    pub(crate) cands: [[Mv; 4]; 2],
    pub(crate) ncands: [usize; 2],
}

/// The analysis's verdict: an inter macroblock (its reconstruction in
/// `rec`, its motion in the state) or an intra one (with the 4x4 modes
/// the walk keeps for its neighbours).
// The intra variant carries the decision the inter ones carry too (of
// similar size, but generic here); boxing buys nothing for a value that
// lives for one macroblock.
#[allow(clippy::large_enum_variant)]
pub(crate) enum MbChoice<D> {
    Inter(D),
    Intra(MbDecision, [u8; 16]),
}

/// One macroblock's samples, all planes, for trying candidates in place.
#[derive(Clone)]
struct MbPix<S: Sample> {
    y: [S; 256],
    c: [[S; 256]; 2],
}

/// Chroma macroblock size `(width, height)`; `(0, 0)` for monochrome.
fn chroma_dims<S: Sample>(ctx: &IntraCtx<S>) -> (usize, usize) {
    if ctx.chroma_h == 0 {
        (0, 0)
    } else if ctx.c444 {
        (16, 16)
    } else {
        (8, ctx.chroma_h)
    }
}

impl<S: Sample> MbPix<S> {
    fn new() -> Self {
        MbPix {
            y: [S::default(); 256],
            c: [[S::default(); 256]; 2],
        }
    }

    fn save(&mut self, ctx: &IntraCtx<S>, rec: &[Recon<S>], mb_x: usize, mb_y: usize) {
        let o = rec[0].offset((mb_x * 16) as isize, (mb_y * 16) as isize);
        gather(&rec[0], o, 16, 16, &mut self.y);
        let (cw, ch) = chroma_dims(ctx);
        for c in 0..2 {
            if cw != 0 {
                let p = &rec[c + 1];
                let o = p.offset((mb_x * cw) as isize, (mb_y * ch) as isize);
                gather(p, o, cw, ch, &mut self.c[c][..cw * ch]);
            }
        }
    }

    fn save_luma(&mut self, rec: &[Recon<S>], mb_x: usize, mb_y: usize) {
        let o = rec[0].offset((mb_x * 16) as isize, (mb_y * 16) as isize);
        gather(&rec[0], o, 16, 16, &mut self.y);
    }

    fn restore_luma(&self, rec: &mut [Recon<S>], mb_x: usize, mb_y: usize) {
        let p = &mut rec[0];
        let o = p.offset((mb_x * 16) as isize, (mb_y * 16) as isize);
        for y in 0..16 {
            p.data[o + y * p.stride..o + y * p.stride + 16]
                .copy_from_slice(&self.y[y * 16..y * 16 + 16]);
        }
    }

    fn restore(&self, ctx: &IntraCtx<S>, rec: &mut [Recon<S>], mb_x: usize, mb_y: usize) {
        self.restore_luma(rec, mb_x, mb_y);
        let (cw, ch) = chroma_dims(ctx);
        for c in 0..2 {
            if cw != 0 {
                let p = &mut rec[c + 1];
                let o = p.offset((mb_x * cw) as isize, (mb_y * ch) as isize);
                for y in 0..ch {
                    p.data[o + y * p.stride..o + y * p.stride + cw]
                        .copy_from_slice(&self.c[c][y * cw..y * cw + cw]);
                }
            }
        }
    }
}

fn ssd_luma<S: Sample>(
    ctx: &IntraCtx<S>,
    rec: &[Recon<S>],
    src: &SrcMb<S>,
    mb_x: usize,
    mb_y: usize,
) -> u64 {
    let (px, py) = (mb_x * 16, mb_y * 16);
    let o = rec[0].offset(px as isize, py as isize);
    (ctx.dist.ssd)(
        &src.y[py * src.ys + px..],
        src.ys,
        &rec[0].data[o..],
        rec[0].stride,
        16,
        16,
    )
}

fn ssd_mb<S: Sample>(
    ctx: &IntraCtx<S>,
    rec: &[Recon<S>],
    src: &SrcMb<S>,
    mb_x: usize,
    mb_y: usize,
) -> u64 {
    let mut d = ssd_luma(ctx, rec, src, mb_x, mb_y);
    let (cw, ch) = chroma_dims(ctx);
    if cw != 0 {
        let (cx, cy) = (mb_x * cw, mb_y * ch);
        for c in 0..2 {
            let p = &rec[c + 1];
            let o = p.offset(cx as isize, cy as isize);
            d += (ctx.dist.ssd)(
                &src.c[c][cy * src.cs + cx..],
                src.cs,
                &p.data[o..],
                p.stride,
                cw,
                ch,
            );
        }
    }
    d
}

/// Whether a residual makes the macroblock carry `mb_qp_delta`, and the
/// delta it then carries from `qp_pred` to `qp`.
fn delta(qp: i32, qp_pred: i32, residual: bool) -> i8 {
    if residual { (qp - qp_pred) as i8 } else { 0 }
}

/// The early-skip threshold on the skip prediction's luma SAD: below it
/// the residual is quantised to see whether anything survives.
fn early_skip_sad(qp: i32) -> u32 {
    let qstep = 0.625 * (qp as f32 / 6.0).exp2();
    (256.0 * qstep * 0.5) as u32
}

/// The early-skip ceiling on the skip prediction's luma squared error:
/// a residual that quantises to nothing can still be one the
/// rate-distortion comparison would rather code.
fn early_skip_ssd(qp: i32, factor: f32) -> f32 {
    let qstep = 0.625 * (qp as f32 / 6.0).exp2();
    256.0 * qstep * qstep * factor
}

// ---------------------------------------------------------------------------
// Intra
// ---------------------------------------------------------------------------

/// A coded intra candidate: the decision and the 4x4 modes it leaves.
struct IntraCand {
    dec: MbDecision,
    modes: [u8; 16],
    /// SATD-scale estimate (prediction SATD plus mode bits).
    est: f32,
}

/// Bits an Intra_4x4 / Intra_8x8 mode costs: one for the predicted mode,
/// four otherwise.
#[inline]
fn mode_bits(mode: u8, predicted: u8) -> f32 {
    if mode == predicted { 1.0 } else { 4.0 }
}

/// Intra_4x4 over the macroblock: per block (decode order), each legal
/// mode scored by SATD plus `lam` times its mode bits, the winner coded
/// and reconstructed before the next block predicts from it. `None` when
/// the running estimate passes `budget`.
#[allow(clippy::too_many_arguments)]
fn i4x4<S: Sample>(
    ctx: &IntraCtx<S>,
    rec: &mut Recon<S>,
    px: usize,
    py: usize,
    src: &[S],
    ss: usize,
    nb: &Nb,
    lam: f32,
    budget: f32,
) -> Option<IntraCand> {
    let mb = nb.avail;
    let base = rec.offset(px as isize, py as isize);
    let mut dec = MbDecision::default();
    let mut chosen = [2u8; 16];
    let mut est = 6.0 * lam; // mb_type and the transform flag, roughly
    let mut pred = [S::default(); 16];
    for scan in 0..16 {
        let bx = crate::h264::tables::BLK4X4_X[scan] as usize;
        let by = crate::h264::tables::BLK4X4_Y[scan] as usize;
        let raster = by * 4 + bx;
        let boff = base + by * 4 * rec.stride + bx * 4;
        let soff = by * 4 * ss + bx * 4;
        let av = avail_4x4(bx, by, mb);
        let left = if bx > 0 {
            Some(chosen[raster - 1])
        } else if mb.left {
            nb.left_modes[by]
        } else {
            None
        };
        let top = if by > 0 {
            Some(chosen[raster - 4])
        } else if mb.top {
            nb.top_modes[bx]
        } else {
            None
        };
        let predicted = predicted_mode(left, top);
        let mut best = (f32::MAX, 2u8);
        for mode in 0..9u8 {
            if predict_4x4(rec, boff, rec.stride, mode, av, ctx.bit_depth).is_err() {
                continue;
            }
            gather(rec, boff, 4, 4, &mut pred);
            let c = (ctx.dist.satd)(&src[soff..], ss, &pred, 4, 4, 4) as f32
                + lam * mode_bits(mode, predicted);
            if c < best.0 {
                best = (c, mode);
            }
        }
        est += best.0;
        if est > budget {
            return None;
        }
        chosen[raster] = best.1;
        let _ = predict_4x4(rec, boff, rec.stride, best.1, av, ctx.bit_depth);
        let (lv, n, _) = code_block_4x4(ctx, rec, boff, &src[soff..], ss, 0, ctx.qp_prime, true);
        dec.luma[raster] = lv;
        dec.nz_luma[raster] = n as u8;
        if n != 0 {
            reconstruct_4x4(ctx, rec, boff, &lv, 0, ctx.qp_prime, None);
        }
        dec.luma_pred[raster] = as_syntax(best.1, predicted);
    }
    dec.kind = MbKind::I4x4;
    for blk8 in 0..4 {
        if quad_rasters(blk8).iter().any(|&r| dec.nz_luma[r] != 0) {
            dec.cbp_luma |= 1 << blk8;
        }
    }
    Some(IntraCand {
        dec,
        modes: chosen,
        est,
    })
}

/// Intra_8x8 over the macroblock, as [`i4x4`] over four quads.
#[allow(clippy::too_many_arguments)]
fn i8x8<S: Sample>(
    ctx: &IntraCtx<S>,
    rec: &mut Recon<S>,
    px: usize,
    py: usize,
    src: &[S],
    ss: usize,
    nb: &Nb,
    lam: f32,
    budget: f32,
) -> Option<IntraCand> {
    let mb = nb.avail;
    let base = rec.offset(px as isize, py as isize);
    let mut dec = MbDecision::default();
    let mut chosen = [2u8; 16];
    let mut est = 6.0 * lam;
    let mut pred = [S::default(); 64];
    for blk8 in 0..4 {
        let (bx, by) = ((blk8 & 1) * 2, (blk8 >> 1) * 2);
        let raster = by * 4 + bx;
        let boff = base + by * 4 * rec.stride + bx * 4;
        let soff = by * 4 * ss + bx * 4;
        let av = avail_8x8(bx, by, mb);
        let left = if bx > 0 {
            Some(chosen[raster - 1])
        } else if mb.left {
            nb.left_modes[by]
        } else {
            None
        };
        let top = if by > 0 {
            Some(chosen[raster - 4])
        } else if mb.top {
            nb.top_modes[bx]
        } else {
            None
        };
        let predicted = predicted_mode(left, top);
        let mut best = (f32::MAX, 2u8);
        for mode in 0..9u8 {
            if predict_8x8(rec, boff, rec.stride, mode, av, ctx.bit_depth).is_err() {
                continue;
            }
            gather(rec, boff, 8, 8, &mut pred);
            let c = (ctx.dist.satd)(&src[soff..], ss, &pred, 8, 8, 8) as f32
                + lam * mode_bits(mode, predicted);
            if c < best.0 {
                best = (c, mode);
            }
        }
        est += best.0;
        if est > budget {
            return None;
        }
        let _ = predict_8x8(rec, boff, rec.stride, best.1, av, ctx.bit_depth);
        let (lv, counts) = code_block_8x8(ctx, rec, boff, &src[soff..], ss, 0, ctx.qp_prime, true);
        reconstruct_8x8(ctx, rec, boff, &lv, 0, ctx.qp_prime);
        dec.luma.as_flattened_mut()[blk8 * 64..blk8 * 64 + 64].copy_from_slice(&lv);
        let syn = as_syntax(best.1, predicted);
        for (sub, &r) in quad_rasters(blk8).iter().enumerate() {
            dec.nz_luma[r] = counts[sub];
            chosen[r] = best.1;
            dec.luma_pred[r] = syn;
        }
        if counts.iter().any(|&n| n != 0) {
            dec.cbp_luma |= 1 << blk8;
        }
    }
    dec.kind = MbKind::I8x8;
    dec.transform_8x8 = true;
    Some(IntraCand {
        dec,
        modes: chosen,
        est,
    })
}

/// The cheapest Intra_16x16 mode by SATD plus mode bits, predicting into
/// `rec` (which the caller then overwrites). `(estimate, mode)`.
fn best_i16<S: Sample>(
    ctx: &IntraCtx<S>,
    rec: &mut Recon<S>,
    px: usize,
    py: usize,
    src: &[S],
    ss: usize,
    mb: MbAvail,
    lam: f32,
) -> (f32, u8) {
    let off = rec.offset(px as isize, py as isize);
    let av = crate::h264::intra::IntraAvail {
        top: mb.top,
        left: mb.left,
        top_left: mb.top_left,
        top_right: false,
    };
    let stride = rec.stride;
    let mut best = (f32::MAX, 2u8);
    for mode in 0..4u8 {
        if predict_16x16(rec, off, stride, mode, av, ctx.bit_depth).is_err() {
            continue;
        }
        let c = (ctx.dist.satd)(src, ss, &rec.data[off..], stride, 16, 16) as f32
            + lam * if mode == 2 { 1.0 } else { 2.0 };
        if c < best.0 {
            best = (c, mode);
        }
    }
    // mb_type with its cbp folded in is a few bins more than I_NxN's.
    (best.0 + 2.0 * lam, best.1)
}

/// The chroma prediction mode by SATD over both components, plus its
/// bits; returns the mode.
fn best_chroma_mode<S: Sample>(
    ctx: &IntraCtx<S>,
    rec: &mut [Recon<S>],
    mb_x: usize,
    mb_y: usize,
    src: &SrcMb<S>,
    mb: MbAvail,
    lam: f32,
) -> u8 {
    let (cw, ch) = (8usize, ctx.chroma_h);
    let (cx, cy) = (mb_x * cw, mb_y * ch);
    let av = crate::h264::intra::IntraAvail {
        top: mb.top,
        left: mb.left,
        top_left: mb.top_left,
        top_right: false,
    };
    let mut best = (f32::MAX, 0u8);
    for mode in 0..4u8 {
        let mut cost = lam * [1.0, 2.0, 3.0, 3.0][mode as usize];
        let mut ok = true;
        for comp in 0..2 {
            let plane = &mut rec[comp + 1];
            let off = plane.offset(cx as isize, cy as isize);
            let stride = plane.stride;
            if predict_chroma(
                plane,
                off,
                stride,
                mode,
                av,
                [mb.left; 4],
                ctx.bit_depth,
                ch,
            )
            .is_err()
            {
                ok = false;
                break;
            }
            cost += (ctx.dist.satd)(
                &src.c[comp][cy * src.cs + cx..],
                src.cs,
                &plane.data[off..],
                stride,
                cw,
                ch,
            ) as f32;
        }
        if ok && cost < best.0 {
            best = (cost, mode);
        }
    }
    best.1
}

/// The cheapest intra estimate this macroblock has — Intra_16x16's —
/// predicting into `rec`. What the inter decisions compare against before
/// paying for a full intra analysis.
pub(crate) fn intra_estimate<S: Sample>(
    ctx: &IntraCtx<S>,
    rec: &mut [Recon<S>],
    mb_x: usize,
    mb_y: usize,
    src: &SrcMb<S>,
    mb: MbAvail,
) -> f32 {
    let (px, py) = (mb_x * 16, mb_y * 16);
    best_i16(
        ctx,
        &mut rec[0],
        px,
        py,
        &src.y[py * src.ys + px..],
        src.ys,
        mb,
        lambda_satd(ctx.qp),
    )
    .0
}

/// Decide and code one macroblock as intra: chroma mode by SATD, the
/// luma types (16x16, 8x8, 4x4) estimated, and the survivors coded and
/// compared by `SSD + lambda * rate`, `rate` the writer's trial cost of
/// the whole macroblock. Leaves the winner's reconstruction in `rec`;
/// returns it with its modes and its cost `J` (luma SSD plus rate —
/// chroma is the same in every candidate).
#[allow(clippy::too_many_arguments)]
pub(crate) fn intra_mb<S: Sample>(
    cfg: &AnalysisCfg,
    ctx: &IntraCtx<S>,
    rec: &mut [Recon<S>],
    mb_x: usize,
    mb_y: usize,
    src: &SrcMb<S>,
    nb: &Nb,
    qp_pred: i32,
    trellis: Option<&Trellis>,
    rate: &mut dyn FnMut(&MbDecision) -> f64,
) -> (MbDecision, [u8; 16], f64) {
    let lam1 = lambda_satd(ctx.qp);
    let lam2 = lambda_ssd(ctx.qp) as f64;
    let (px, py) = (mb_x * 16, mb_y * 16);
    if ctx.c444 {
        // 4:4:4 codes its chroma luma-style with the luma's modes; the
        // established decision handles that coupling.
        let (mut dec, modes) = code_macroblock_modes8(
            ctx,
            rec,
            mb_x,
            mb_y,
            src.y,
            src.ys,
            src.c,
            src.cs,
            nb.avail,
            nb.left_modes,
            nb.top_modes,
            nb.left_modes,
        );
        let residual = dec.kind == MbKind::I16x16 || dec.cbp_luma != 0 || dec.cbp_chroma != 0;
        dec.qp_delta = delta(ctx.qp, qp_pred, residual);
        let j = ssd_luma(ctx, rec, src, mb_x, mb_y) as f64 + lam2 * rate(&dec);
        return (dec, modes, j);
    }
    // Chroma first: one mode for every luma candidate.
    let mut chroma = MbDecision::default();
    if ctx.chroma_h != 0 {
        let mode = best_chroma_mode(ctx, rec, mb_x, mb_y, src, nb.avail, lam1);
        let (cx, cy) = (mb_x * 8, mb_y * ctx.chroma_h);
        let co = cy * src.cs + cx;
        code_chroma(
            ctx,
            rec,
            cx,
            cy,
            &[&src.c[0][co..], &src.c[1][co..]],
            src.cs,
            mode,
            nb.avail,
            &mut chroma,
        );
    }
    let sy = &src.y[py * src.ys + px..];
    let (e16, m16) = best_i16(ctx, &mut rec[0], px, py, sy, src.ys, nb.avail, lam1);

    let with_chroma = |d: &mut MbDecision| {
        d.chroma_mode = chroma.chroma_mode;
        d.cbp_chroma = chroma.cbp_chroma;
        d.chroma_dc = chroma.chroma_dc;
        d.chroma_ac = chroma.chroma_ac;
        d.nz_chroma = chroma.nz_chroma;
        let residual = d.kind == MbKind::I16x16 || d.cbp_luma != 0 || d.cbp_chroma != 0;
        d.qp_delta = delta(ctx.qp, qp_pred, residual);
    };

    let mut best: Option<(f64, MbDecision, [u8; 16])> = None;
    let mut best_pix = MbPix::new();
    let use_rate = cfg.rd;
    let mut consider = |d: MbDecision,
                        modes: [u8; 16],
                        est: f32,
                        rec: &mut [Recon<S>],
                        rate: &mut dyn FnMut(&MbDecision) -> f64| {
        // Without rate-distortion decisions, the estimates decide.
        let j = if use_rate {
            ssd_luma(ctx, rec, src, mb_x, mb_y) as f64 + lam2 * rate(&d)
        } else {
            est as f64
        };
        if best.as_ref().is_none_or(|b| j < b.0) {
            best_pix.save_luma(rec, mb_x, mb_y);
            best = Some((j, d, modes));
        }
    };

    let mut any = false;
    let budget = e16 * cfg.intra_ratio.max(1.0) * 1.15;
    let mut floor = e16;
    if ctx.t8x8
        && let Some(mut c) = i8x8(ctx, &mut rec[0], px, py, sy, src.ys, nb, lam1, budget)
    {
        floor = floor.min(c.est);
        with_chroma(&mut c.dec);
        consider(c.dec, c.modes, c.est, rec, rate);
        any = true;
    }
    if cfg.i4x4 {
        let b = budget.min(floor * 1.15);
        if let Some(mut c) = i4x4(ctx, &mut rec[0], px, py, sy, src.ys, nb, lam1, b) {
            floor = floor.min(c.est);
            with_chroma(&mut c.dec);
            consider(c.dec, c.modes, c.est, rec, rate);
            any = true;
        }
    }
    if e16 <= floor * 1.15 || !any {
        let mut d = MbDecision::default();
        code_i16x16(ctx, &mut rec[0], px, py, sy, src.ys, m16, nb.avail, &mut d);
        with_chroma(&mut d);
        consider(d, [2u8; 16], e16, rec, rate);
    }
    let (j, mut dec, modes) = best.expect("Intra_16x16 is always a candidate");
    match trellis {
        Some(t) => {
            let tctx = IntraCtx {
                trellis: Some(t),
                ..*ctx
            };
            requant_intra(&tctx, rec, mb_x, mb_y, src, nb, &mut dec, &modes, qp_pred);
        }
        None => best_pix.restore_luma(rec, mb_x, mb_y),
    }
    (dec, modes, j)
}

/// Code an intra decision again at its chosen modes under `ctx` (the
/// trellis's): luma by type, chroma at its mode, the coded block patterns
/// and the delta re-derived. The reconstruction lands in `rec`.
#[allow(clippy::too_many_arguments)]
fn requant_intra<S: Sample>(
    ctx: &IntraCtx<S>,
    rec: &mut [Recon<S>],
    mb_x: usize,
    mb_y: usize,
    src: &SrcMb<S>,
    nb: &Nb,
    dec: &mut MbDecision,
    modes: &[u8; 16],
    qp_pred: i32,
) {
    let (px, py) = (mb_x * 16, mb_y * 16);
    let sy = &src.y[py * src.ys + px..];
    match dec.kind {
        MbKind::I16x16 => {
            let mut d = MbDecision::default();
            code_i16x16(
                ctx,
                &mut rec[0],
                px,
                py,
                sy,
                src.ys,
                dec.intra16_mode,
                nb.avail,
                &mut d,
            );
            dec.luma = d.luma;
            dec.luma_dc = d.luma_dc;
            dec.nz_luma = d.nz_luma;
            dec.cbp_luma = d.cbp_luma;
        }
        MbKind::I4x4 | MbKind::I8x8 => {
            let (levels, nz) = if dec.kind == MbKind::I4x4 {
                code_i4x4_plane_fixed(
                    ctx,
                    &mut rec[0],
                    px,
                    py,
                    sy,
                    src.ys,
                    nb.avail,
                    modes,
                    0,
                    ctx.qp_prime,
                )
            } else {
                code_i8x8_plane_fixed(
                    ctx,
                    &mut rec[0],
                    px,
                    py,
                    sy,
                    src.ys,
                    nb.avail,
                    modes,
                    0,
                    ctx.qp_prime,
                )
            };
            dec.luma = levels;
            dec.nz_luma = nz;
            dec.cbp_luma = 0;
            for blk8 in 0..4 {
                if quad_rasters(blk8).iter().any(|&r| nz[r] != 0) {
                    dec.cbp_luma |= 1 << blk8;
                }
            }
        }
    }
    if ctx.chroma_h != 0 {
        let (cx, cy) = (mb_x * 8, mb_y * ctx.chroma_h);
        let co = cy * src.cs + cx;
        let mut c = MbDecision::default();
        code_chroma(
            ctx,
            rec,
            cx,
            cy,
            &[&src.c[0][co..], &src.c[1][co..]],
            src.cs,
            dec.chroma_mode,
            nb.avail,
            &mut c,
        );
        dec.cbp_chroma = c.cbp_chroma;
        dec.chroma_dc = c.chroma_dc;
        dec.chroma_ac = c.chroma_ac;
        dec.nz_chroma = c.nz_chroma;
    }
    let residual = dec.kind == MbKind::I16x16 || dec.cbp_luma != 0 || dec.cbp_chroma != 0;
    dec.qp_delta = delta(ctx.qp, qp_pred, residual);
}

// ---------------------------------------------------------------------------
// P macroblocks
// ---------------------------------------------------------------------------

/// A searched P shape.
#[derive(Clone, Copy)]
struct PCand {
    kind: InterMbKind,
    sub_shape: [SubMbShape; 4],
    mvs: [Mv; 16],
    mvds: [Mv; 16],
    /// List-0 reference index per 8x8.
    refs: [i8; 4],
    est: f32,
}

impl PCand {
    fn fill(&mut self, (x, y, w, h): (usize, usize, usize, usize), mv: Mv, mvd: Mv) {
        for by in y / 4..(y + h) / 4 {
            for bx in x / 4..(x + w) / 4 {
                self.mvs[by * 4 + bx] = mv;
                self.mvds[by * 4 + bx] = mvd;
            }
        }
    }

    fn rects(&self, out: &mut [(usize, usize, usize, usize); 16]) -> usize {
        let d = InterDecision {
            kind: self.kind,
            sub_shape: self.sub_shape,
            ..InterDecision::default()
        };
        d.rects(out)
    }
}

#[inline]
fn sub(a: Mv, b: Mv) -> Mv {
    Mv::new(a.x - b.x, a.y - b.y)
}

/// Decide and code one macroblock of a P picture. See the module docs;
/// `rate` is the writer's trial cost of a candidate in the current
/// entropy state, `qp_pred` the quantiser a `mb_qp_delta` is coded
/// against.
#[allow(clippy::too_many_arguments)]
pub(crate) fn p_mb<S: Sample>(
    an: &mut PicAn<'_, S>,
    ctx: &IntraCtx<S>,
    rec: &mut [Recon<S>],
    pref: &PRef<S>,
    more: &[&[Recon<S>]],
    mb_x: usize,
    mb_y: usize,
    src: &SrcMb<S>,
    st: &mut MbMotionState,
    nb: &Nb,
    qp_pred: i32,
    rate: &mut dyn FnMut(PMb<'_>) -> f64,
) -> MbChoice<InterDecision> {
    an.prepare(ctx.qp);
    let cfg = an.cfg;
    let (px, py) = (mb_x * 16, mb_y * 16);
    let lam1 = lambda_satd(ctx.qp);
    let lam2 = lambda_ssd(ctx.qp) as f64;
    let planes2 = [pref.planes, pref.planes];
    st.reset_mb();
    let pmv = st.predict(0, 0, 0, 0, 16, 16);
    let skip_mv = st.skip_mv();
    let skip_dec = InterDecision {
        kind: InterMbKind::PSkip,
        ..InterDecision::default()
    };
    let sy = &src.y[py * src.ys + px..];
    let predict_r =
        |rec: &mut [Recon<S>], x: usize, y: usize, w: usize, h: usize, mv: Mv, r: i8| {
            let pl = if r == 0 {
                pref.planes
            } else {
                more[r as usize - 1]
            };
            predict_inter_rect(
                ctx,
                rec,
                [pl, pl],
                px + x,
                py + y,
                w,
                h,
                [true, false],
                [mv, Mv::ZERO],
                pref.weighting,
            );
        };
    let predict = |rec: &mut [Recon<S>], x: usize, y: usize, w: usize, h: usize, mv: Mv| {
        predict_inter_rect(
            ctx,
            rec,
            planes2,
            px + x,
            py + y,
            w,
            h,
            [true, false],
            [mv, Mv::ZERO],
            pref.weighting,
        );
    };

    // Skip, before anything is searched.
    predict(rec, 0, 0, 16, 16, skip_mv);
    if cfg.early_skip {
        let o = rec[0].offset(px as isize, py as isize);
        let sad = (ctx.dist.sad)(sy, src.ys, &rec[0].data[o..], rec[0].stride, 16, 16);
        if sad < early_skip_sad(ctx.qp) {
            let r = code_inter_mb_residual(
                ctx,
                rec,
                mb_x,
                mb_y,
                src.y,
                src.ys,
                src.c,
                src.cs,
                false,
                Some(false),
            );
            if r.cbp_luma == 0
                && r.cbp_chroma == 0
                && (ssd_luma(ctx, rec, src, mb_x, mb_y) as f32)
                    < early_skip_ssd(ctx.qp, cfg.skip_ssd)
            {
                st.commit_part(0, 0, 16, 16, [Some(skip_mv), None]);
                return MbChoice::Inter(skip_dec);
            }
            predict(rec, 0, 0, 16, 16, skip_mv);
        }
    }
    let skip_ssd = ssd_mb(ctx, rec, src, mb_x, mb_y);
    let skip_est = if cfg.rd {
        0.0
    } else {
        let o = rec[0].offset(px as isize, py as isize);
        (ctx.dist.satd)(sy, src.ys, &rec[0].data[o..], rec[0].stride, 16, 16) as f32
    };

    // The searches.
    let me = an.me[0]
        .as_ref()
        .expect("a P picture has a list-0 reference");
    let costs = an.cost(ctx.qp);
    let vr = ctx.motion.vertical_search(ctx.field);
    let mut cands = [Mv::ZERO; 8];
    let mut nc = 0;
    for &c in [skip_mv, Mv::ZERO]
        .iter()
        .chain(nb.cands[0][..nb.ncands[0]].iter())
    {
        if nc < cands.len() {
            cands[nc] = c;
            nc += 1;
        }
    }
    let nref = 1 + more.len();
    let run_r =
        |r: usize, x: usize, y: usize, w: usize, h: usize, pred: Mv, cands: &[Mv], mc: &MeCfg| {
            let s = Search {
                dist: ctx.dist,
                r: if r == 0 { me } else { &an.me_more[r - 1] },
                src: &sy[y * src.ys + x..],
                src_stride: src.ys,
                x: (px + x) as i32,
                y: (py + y) as i32,
                w,
                h,
                pred,
                cost: costs,
                bounds: search_bounds(pref.search, (px + x) as i32, (py + y) as i32, w, h, vr),
            };
            s.run(cands, mc)
        };
    // 16x16 against every active reference, each at its own predictor and
    // charged its index (te(v): one bit over two entries, else ue(v)).
    let ref_bits = |r: usize| -> f32 {
        if nref == 1 {
            0.0
        } else if nref == 2 {
            1.0
        } else {
            (2 * (32 - (r as u32 + 1).leading_zeros()) - 1) as f32
        }
    };
    let (mut mv16, mut c16, mut pmv16, mut r16) = (Mv::ZERO, u32::MAX, pmv, 0usize);
    // Each reference's own 16x16 vector and cost, for the partitions.
    let mut mv16r = [Mv::ZERO; 16];
    let mut c16r = [u32::MAX; 16];
    for r in 0..nref {
        let p = if r == 0 {
            pmv
        } else {
            st.predict(0, r as i8, 0, 0, 16, 16)
        };
        if r > 0 && nc < cands.len() {
            cands[nc] = p;
        }
        let k = if r > 0 { (nc + 1).min(cands.len()) } else { nc };
        let (m, _, c) = run_r(r, 0, 0, 16, 16, p, &cands[..k], &cfg.me);
        let c = c + (lam1 * ref_bits(r)) as u32;
        if r < mv16r.len() {
            (mv16r[r], c16r[r]) = (m, c);
        }
        if c < c16 {
            (mv16, c16, pmv16, r16) = (m, c, p, r);
        }
    }
    let rr = r16 as i8;
    let mut c16v = PCand {
        kind: InterMbKind::P16x16,
        sub_shape: [SubMbShape::S8x8; 4],
        mvs: [mv16; 16],
        mvds: [sub(mv16, pmv16); 16],
        refs: [rr; 4],
        est: c16 as f32 + lam1,
    };
    c16v.fill((0, 0, 16, 16), mv16, sub(mv16, pmv16));
    let mut shapes: [Option<PCand>; 4] = [Some(c16v), None, None, None];
    if ctx.subparts {
        // Each partition chooses its own reference among those whose 16x16
        // search came within a margin of the best one's (a reference far
        // behind at 16x16 rarely wins a part), at its own predictor and
        // charged its index.
        let mut near = [0usize; 16];
        let mut nn = 0;
        for r in 0..nref.min(mv16r.len()) {
            if r == r16 || (c16r[r] as f32) < c16 as f32 * cfg.mixed_refs {
                near[nn] = r;
                nn += 1;
            }
        }
        let near = &near[..nn];
        let part_search = |st: &MbMotionState,
                           (x, y, w, h): (usize, usize, usize, usize),
                           extra: &[Mv]|
         -> (Mv, Mv, u32, i8) {
            let mut best = (Mv::ZERO, Mv::ZERO, u32::MAX, 0i8);
            for &r in near {
                let p = st.predict(0, r as i8, x, y, w, h);
                let mut cands = [Mv::ZERO; 6];
                let mut nc = 0;
                for &c in [mv16r[r], p, Mv::ZERO].iter().chain(extra) {
                    if nc < cands.len() {
                        cands[nc] = c;
                        nc += 1;
                    }
                }
                let (mv, _, c) = run_r(r, x, y, w, h, p, &cands[..nc], &cfg.me_sub);
                let c = c + (lam1 * ref_bits(r)) as u32;
                if c < best.2 {
                    best = (mv, p, c, r as i8);
                }
            }
            best
        };
        // 8x8 first; the two-partition shapes only when it beat 16x16.
        st.reset_mb();
        let mut t = PCand {
            kind: InterMbKind::P8x8,
            est: lam1 * 7.0,
            ..c16v
        };
        for part in 0..4 {
            let (x, y) = ((part & 1) * 8, (part >> 1) * 8);
            let (mv, p, c, r) = part_search(st, (x, y, 8, 8), &[]);
            t.est += c as f32;
            t.fill((x, y, 8, 8), mv, sub(mv, p));
            t.refs[part] = r;
            st.commit_part_ref(x, y, 8, 8, mv, r);
        }
        let c8 = t.est;
        shapes[3] = Some(t);
        if c8 < c16v.est {
            for (i, kind) in [(1, InterMbKind::P16x8), (2, InterMbKind::P8x16)] {
                st.reset_mb();
                let mut c = PCand {
                    kind,
                    est: lam1 * 3.0,
                    ..c16v
                };
                for &(x, y, w, h) in kind.parts() {
                    let m8a = t.mvs[(y / 4) * 4 + x / 4];
                    let m8b = t.mvs[((y + h) / 4 - 1) * 4 + (x + w) / 4 - 1];
                    let (mv, p, cc, r) = part_search(st, (x, y, w, h), &[m8a, m8b]);
                    c.est += cc as f32;
                    c.fill((x, y, w, h), mv, sub(mv, p));
                    for py8 in y / 8..(y + h) / 8 {
                        for px8 in x / 8..(x + w) / 8 {
                            c.refs[py8 * 2 + px8] = r;
                        }
                    }
                    st.commit_part_ref(x, y, w, h, mv, r);
                }
                shapes[i] = Some(c);
            }
        }
    }
    let best_est = shapes
        .iter()
        .flatten()
        .map(|c| c.est)
        .fold(f32::MAX, f32::min);

    // Intra: estimated cheaply first, analysed only when it might win.
    let i_est = intra_estimate(ctx, rec, mb_x, mb_y, src, nb.avail);

    if !cfg.rd {
        // Decided on the estimates; only the winner is coded.
        let best = *shapes
            .iter()
            .flatten()
            .min_by(|a, b| a.est.total_cmp(&b.est))
            .expect("16x16 is always searched");
        let inter = best.est.min(skip_est);
        if i_est < inter * cfg.intra_ratio {
            let (idec, modes, e) = intra_mb(
                &cfg,
                ctx,
                rec,
                mb_x,
                mb_y,
                src,
                nb,
                qp_pred,
                None,
                &mut |_: &MbDecision| 0.0,
            );
            if (e as f32) < inter {
                st.reset_mb();
                return MbChoice::Intra(idec, modes);
            }
        }
        st.reset_mb();
        if skip_est <= best.est {
            st.commit_part(0, 0, 16, 16, [Some(skip_mv), None]);
            predict(rec, 0, 0, 16, 16, skip_mv);
            return MbChoice::Inter(skip_dec);
        }
        let mut rects = [(0usize, 0usize, 0usize, 0usize); 16];
        let n = best.rects(&mut rects);
        for &(x, y, w, h) in rects.iter().take(n) {
            let mv = best.mvs[(y / 4) * 4 + x / 4];
            let r = best.refs[crate::h264::cavlc::part_index_of(x, y)];
            st.commit_part_ref(x, y, w, h, mv, r);
            predict_r(rec, x, y, w, h, mv, r);
        }
        let allow8 = ctx.t8x8
            && (best.kind != InterMbKind::P8x8 || best.sub_shape.iter().all(|s| s.count() == 1));
        let r = code_inter_mb_residual(
            ctx,
            rec,
            mb_x,
            mb_y,
            src.y,
            src.ys,
            src.c,
            src.cs,
            allow8,
            Some(allow8),
        );
        let residual = r.cbp_luma != 0 || r.cbp_chroma != 0;
        let kind = if best.kind == InterMbKind::P16x16
            && !residual
            && best.mvs[0] == skip_mv
            && best.refs[0] == 0
        {
            InterMbKind::PSkip
        } else {
            best.kind
        };
        return MbChoice::Inter(InterDecision {
            kind,
            transform_8x8: r.transform_8x8,
            mvd: best.mvds,
            sub_shape: best.sub_shape,
            ref_idx: 0,
            refs: best.refs,
            cbp_luma: r.cbp_luma,
            cbp_chroma: r.cbp_chroma,
            qp_delta: delta(ctx.qp, qp_pred, residual),
            luma: r.luma,
            chroma_dc: r.chroma_dc,
            chroma_ac: r.chroma_ac,
            nz_luma: r.nz_luma,
            nz_chroma: r.nz_chroma,
        });
    }

    // The comparison, in coded form (by the trellis where it decides).
    let rctx_store;
    let rctx: &IntraCtx<S> = match an.trellis().filter(|_| cfg.trellis_rd && !ctx.c444) {
        Some(t) => {
            rctx_store = IntraCtx {
                trellis: Some(t),
                ..*ctx
            };
            &rctx_store
        }
        None => ctx,
    };
    let mut best_j = skip_ssd as f64 + lam2 * rate(PMb::Skip(&skip_dec));
    let mut best: MbChoice<InterDecision> = MbChoice::Inter(skip_dec);
    let mut best_pix = MbPix::new();
    let mut best_st = *st;
    best_st.reset_mb();
    best_st.commit_part(0, 0, 16, 16, [Some(skip_mv), None]);
    // Skip's samples are the prediction; re-made at the end if it wins.
    let mut skip_won = true;

    let mut pred_pix = MbPix::new();
    let mut best_pred = MbPix::new();
    let mut best_t8 = (false, false);
    let mut order: Vec<&PCand> = shapes.iter().flatten().collect();
    order.sort_by(|a, b| a.est.total_cmp(&b.est));
    for cand in order.into_iter().take(cfg.rd_max) {
        if cand.est > best_est * cfg.rd_ratio {
            continue;
        }
        st.reset_mb();
        let mut rects = [(0usize, 0usize, 0usize, 0usize); 16];
        let n = cand.rects(&mut rects);
        for &(x, y, w, h) in rects.iter().take(n) {
            let mv = cand.mvs[(y / 4) * 4 + x / 4];
            let r = cand.refs[crate::h264::cavlc::part_index_of(x, y)];
            st.commit_part_ref(x, y, w, h, mv, r);
            predict_r(rec, x, y, w, h, mv, r);
        }
        let allow8 = ctx.t8x8
            && (cand.kind != InterMbKind::P8x8 || cand.sub_shape.iter().all(|s| s.count() == 1));
        let tries: &[bool] = if allow8 { &[false, true] } else { &[false] };
        pred_pix.save(ctx, rec, mb_x, mb_y);
        let mut first: Option<([[i16; 16]; 2], [[[i16; 16]; 16]; 2], [[u8; 16]; 2], u8)> = None;
        for (k, &t8) in tries.iter().enumerate() {
            if k > 0 {
                if ctx.c444 {
                    pred_pix.restore(ctx, rec, mb_x, mb_y);
                } else if first.is_none() {
                    break;
                } else {
                    // The second transform: luma again, the chroma kept.
                    pred_pix.restore_luma(rec, mb_x, mb_y);
                }
            }
            let mut r = code_inter_mb_residual_parts(
                rctx,
                rec,
                mb_x,
                mb_y,
                src.y,
                src.ys,
                src.c,
                src.cs,
                allow8,
                Some(t8),
                k == 0 || ctx.c444,
            );
            if k > 0 && !ctx.c444 {
                if let Some((dc, ac, nz, cbp)) = first {
                    (r.chroma_dc, r.chroma_ac, r.nz_chroma, r.cbp_chroma) = (dc, ac, nz, cbp);
                }
            } else if k == 0 {
                first = Some((r.chroma_dc, r.chroma_ac, r.nz_chroma, r.cbp_chroma));
                if allow8 && r.cbp_luma == 0 && !ctx.c444 {
                    // Nothing survived the 4x4 transform; the 8x8 one
                    // would code (almost always) the same nothing.
                    first = None;
                }
            }
            if t8 && r.cbp_luma == 0 {
                continue; // the same macroblock as the 4x4 one
            }
            let residual = r.cbp_luma != 0 || r.cbp_chroma != 0;
            if cand.kind == InterMbKind::P16x16
                && !residual
                && cand.mvs[0] == skip_mv
                && cand.refs[0] == 0
            {
                continue; // skip, at a higher price
            }
            let dec = InterDecision {
                kind: cand.kind,
                transform_8x8: r.transform_8x8,
                mvd: cand.mvds,
                sub_shape: cand.sub_shape,
                ref_idx: 0,
                refs: cand.refs,
                cbp_luma: r.cbp_luma,
                cbp_chroma: r.cbp_chroma,
                qp_delta: delta(ctx.qp, qp_pred, residual),
                luma: r.luma,
                chroma_dc: r.chroma_dc,
                chroma_ac: r.chroma_ac,
                nz_luma: r.nz_luma,
                nz_chroma: r.nz_chroma,
            };
            let j = ssd_mb(ctx, rec, src, mb_x, mb_y) as f64 + lam2 * rate(PMb::Coded(&dec));
            if j < best_j {
                best_j = j;
                best = MbChoice::Inter(dec);
                best_pix.save(ctx, rec, mb_x, mb_y);
                best_pred.clone_from(&pred_pix);
                best_t8 = (allow8, t8);
                best_st = *st;
                skip_won = false;
            }
        }
    }

    if i_est < best_est * cfg.intra_ratio {
        let (idec, modes, j) = intra_mb(
            &cfg,
            rctx,
            rec,
            mb_x,
            mb_y,
            src,
            nb,
            qp_pred,
            None,
            &mut |d: &MbDecision| rate(PMb::Intra(d)),
        );
        // `j` prices luma distortion only; chroma's must join it to be
        // compared with the inter costs.
        let j =
            j + (ssd_mb(ctx, rec, src, mb_x, mb_y) - ssd_luma(ctx, rec, src, mb_x, mb_y)) as f64;
        if j < best_j {
            st.reset_mb();
            let mut idec = idec;
            if let Some(t) = an.trellis().filter(|_| !ctx.c444 && !cfg.trellis_rd) {
                let tctx = IntraCtx {
                    trellis: Some(t),
                    ..*ctx
                };
                requant_intra(&tctx, rec, mb_x, mb_y, src, nb, &mut idec, &modes, qp_pred);
            }
            return MbChoice::Intra(idec, modes);
        }
    }

    *st = best_st;
    if skip_won {
        predict(rec, 0, 0, 16, 16, skip_mv);
        return best;
    }
    if let (Some(t), MbChoice::Inter(dec)) = (
        an.trellis().filter(|_| !ctx.c444 && !cfg.trellis_rd),
        &mut best,
    ) {
        // The winner's residual again, by the trellis, from its prediction.
        best_pred.restore(ctx, rec, mb_x, mb_y);
        let tctx = IntraCtx {
            trellis: Some(t),
            ..*ctx
        };
        let r = code_inter_mb_residual(
            &tctx,
            rec,
            mb_x,
            mb_y,
            src.y,
            src.ys,
            src.c,
            src.cs,
            best_t8.0,
            Some(best_t8.1),
        );
        let residual = r.cbp_luma != 0 || r.cbp_chroma != 0;
        dec.transform_8x8 = r.transform_8x8;
        dec.cbp_luma = r.cbp_luma;
        dec.cbp_chroma = r.cbp_chroma;
        dec.luma = r.luma;
        dec.nz_luma = r.nz_luma;
        dec.chroma_dc = r.chroma_dc;
        dec.chroma_ac = r.chroma_ac;
        dec.nz_chroma = r.nz_chroma;
        dec.qp_delta = delta(ctx.qp, qp_pred, residual);
        if dec.kind == InterMbKind::P16x16
            && !residual
            && st.motion()[0][0].mv == skip_mv
            && dec.refs[0] == 0
        {
            // Nothing left to say: the macroblock is P_Skip.
            dec.kind = InterMbKind::PSkip;
        }
    } else {
        best_pix.restore(ctx, rec, mb_x, mb_y);
    }
    best
}

// ---------------------------------------------------------------------------
// B macroblocks
// ---------------------------------------------------------------------------

const DIRS: [u8; 3] = [PRED_L0, PRED_L1, PRED_BI];

fn lists(dir: u8) -> [bool; 2] {
    [dir & PRED_L0 != 0, dir & PRED_L1 != 0]
}

/// A searched B shape.
#[derive(Clone, Copy)]
struct BCand {
    kind: BMbKind,
    dir: [u8; 4],
    sub_shape: [SubMbShape; 4],
    mv: [[Mv; 2]; 16],
    mvd: [[Mv; 16]; 2],
    est: f32,
}

impl BCand {
    fn new(kind: BMbKind) -> Self {
        BCand {
            kind,
            dir: [PRED_L0; 4],
            sub_shape: [SubMbShape::S8x8; 4],
            mv: [[Mv::ZERO; 2]; 16],
            mvd: [[Mv::ZERO; 16]; 2],
            est: 0.0,
        }
    }

    fn fill(
        &mut self,
        (x, y, w, h): (usize, usize, usize, usize),
        dir: u8,
        mv: [Mv; 2],
        mvd: [Mv; 2],
    ) {
        let used = lists(dir);
        for by in y / 4..(y + h) / 4 {
            for bx in x / 4..(x + w) / 4 {
                let b = by * 4 + bx;
                self.mv[b] = mv;
                for l in 0..2 {
                    self.mvd[l][b] = if used[l] { mvd[l] } else { Mv::ZERO };
                }
            }
        }
        for py in y / 8..(y + h).div_ceil(8) {
            for px in x / 8..(x + w).div_ceil(8) {
                self.dir[py * 2 + px] = dir;
            }
        }
    }

    fn decision(&self) -> BDecision {
        let mut d = BDecision {
            kind: self.kind,
            dir: self.dir,
            sub_shape: self.sub_shape,
            mv: self.mv,
            mvd: self.mvd,
            ..BDecision::default()
        };
        for part in 0..4 {
            let u = d.used(part);
            d.ref_idx[part] = [if u[0] { 0 } else { -1 }, if u[1] { 0 } else { -1 }];
        }
        d
    }
}

/// The bins a B `mb_type` costs, roughly (Table 9-37's binarisation).
fn b_type_bits(kind: BMbKind, dirs: (u8, u8)) -> f32 {
    match kind {
        BMbKind::BDirect16 => 1.0,
        BMbKind::B16 => {
            if dirs.0 == PRED_BI {
                5.0
            } else {
                3.0
            }
        }
        BMbKind::B16x8 | BMbKind::B8x16 => 6.0,
        _ => 6.0,
    }
}

/// Decide and code one macroblock of a B picture; see [`p_mb`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn b_mb<S: Sample>(
    an: &mut PicAn<'_, S>,
    ctx: &IntraCtx<S>,
    rec: &mut [Recon<S>],
    refs: &BRefs<S>,
    mb_x: usize,
    mb_y: usize,
    src: &SrcMb<S>,
    st: &mut MbMotionState,
    col: &Colocated,
    addr: usize,
    nb: &Nb,
    qp_pred: i32,
    rate: &mut dyn FnMut(BMb<'_>) -> f64,
) -> MbChoice<BDecision> {
    an.prepare(ctx.qp);
    let cfg = an.cfg;
    let (px, py) = (mb_x * 16, mb_y * 16);
    let lam1 = lambda_satd(ctx.qp);
    let lam2 = lambda_ssd(ctx.qp) as f64;
    let sy = &src.y[py * src.ys + px..];
    st.reset_mb();
    let (dref, dmv) = spatial_direct(st, col, addr, false, col.map.cur_parity);
    let dused = [dref[0] >= 0, dref[1] >= 0];
    let ddir = (dused[0] as u8) * PRED_L0 + (dused[1] as u8) * PRED_L1;
    let mut direct = BCand::new(BMbKind::BDirect16);
    for part in 0..4 {
        let r = ((part & 1) * 8, (part >> 1) * 8, 8, 8);
        direct.fill(r, ddir, dmv[part], [Mv::ZERO; 2]);
    }
    let me = [an.me[0].as_ref(), an.me[1].as_ref()];
    let costs = an.cost(ctx.qp);

    // Luma SATD of a rectangle under `dir` at `mv`: through the planes
    // where the prediction is plain, the decoder's kernels where weighted.
    let satd = |(x, y, w, h): (usize, usize, usize, usize), dir: u8, mv: [Mv; 2]| -> u32 {
        let used = lists(dir);
        let wt = refs.weighting_for(used);
        let s = &sy[y * src.ys + x..];
        // The planes hold every vector a search can reach from this block;
        // a derived one (direct's) may lie outside, and goes through the
        // decoder's clamped kernels instead.
        let (ax, ay) = ((px + x) as i32, (py + y) as i32);
        let inside = |l: usize| {
            let (lx, hx, ly, hy) = search_bounds(refs.search[l], ax, ay, w, h, i32::MAX);
            let (mx, my) = (mv[l].x as i32 >> 2, mv[l].y as i32 >> 2);
            !used[l] || (mx >= lx && mx < hx && my >= ly && my < hy)
        };
        if matches!(wt, Weighting::Default) && inside(0) && inside(1) {
            let mut b0 = [S::default(); 256];
            let mut b1 = [S::default(); 256];
            if used[0] && used[1] {
                let (p0, s0) = me[0].unwrap().pred(ax, ay, mv[0], w, h, &mut b0);
                let (p1, s1) = me[1].unwrap().pred(ax, ay, mv[1], w, h, &mut b1);
                let mut avg = [S::default(); 256];
                avg_into(&mut avg, 16, p0, s0, p1, s1, w, h);
                return (ctx.dist.satd)(s, src.ys, &avg, 16, w, h);
            }
            let l = if used[0] { 0 } else { 1 };
            let (p, ps) = me[l].unwrap().pred(ax, ay, mv[l], w, h, &mut b0);
            return (ctx.dist.satd)(s, src.ys, p, ps, w, h);
        }
        rect_satd_under(
            ctx,
            refs,
            (px + x) as i32,
            (py + y) as i32,
            w,
            h,
            s,
            src.ys,
            used,
            mv,
            wt,
        )
    };
    let predict_cand = |rec: &mut [Recon<S>], st: &mut MbMotionState, c: &BCand| {
        st.reset_mb();
        let mut rects = [(0usize, 0usize, 0usize, 0usize); 16];
        let d = c.decision();
        let n = d.rects(&mut rects);
        for &(x, y, w, h) in rects.iter().take(n) {
            let used = d.used(crate::h264::cavlc::part_index_of(x, y));
            let mv = c.mv[(y / 4) * 4 + x / 4];
            st.commit_part(
                x,
                y,
                w,
                h,
                [used[0].then_some(mv[0]), used[1].then_some(mv[1])],
            );
            predict_inter_rect(
                ctx,
                rec,
                refs.planes,
                px + x,
                py + y,
                w,
                h,
                used,
                mv,
                refs.weighting_for(used),
            );
        }
    };

    // Direct, and B_Skip straight away when its prediction leaves nothing.
    let mut dsatd = 0u32;
    for part in 0..4 {
        let r = ((part & 1) * 8, (part >> 1) * 8, 8, 8);
        dsatd += satd(r, ddir, dmv[part]);
    }
    direct.est = dsatd as f32 + lam1;
    predict_cand(rec, st, &direct);
    let skip_dec = BDecision {
        kind: BMbKind::BSkip,
        ..direct.decision()
    };
    let direct_st = *st;
    if cfg.early_skip {
        let o = rec[0].offset(px as isize, py as isize);
        let sad = (ctx.dist.sad)(sy, src.ys, &rec[0].data[o..], rec[0].stride, 16, 16);
        if sad < early_skip_sad(ctx.qp) {
            let r = code_inter_mb_residual(
                ctx,
                rec,
                mb_x,
                mb_y,
                src.y,
                src.ys,
                src.c,
                src.cs,
                false,
                Some(false),
            );
            if r.cbp_luma == 0
                && r.cbp_chroma == 0
                && (ssd_luma(ctx, rec, src, mb_x, mb_y) as f32)
                    < early_skip_ssd(ctx.qp, cfg.skip_ssd)
            {
                return MbChoice::Inter(skip_dec);
            }
            predict_cand(rec, st, &direct);
        }
    }
    let skip_ssd = ssd_mb(ctx, rec, src, mb_x, mb_y);

    // 16x16 per list, then the pair.
    let vr = ctx.motion.vertical_search(ctx.field);
    let search =
        |l: usize, x: usize, y: usize, w: usize, h: usize, pred: Mv, cands: &[Mv], mc: &MeCfg| {
            let s = Search {
                dist: ctx.dist,
                r: me[l].expect("both lists searched"),
                src: &sy[y * src.ys + x..],
                src_stride: src.ys,
                x: (px + x) as i32,
                y: (py + y) as i32,
                w,
                h,
                pred,
                cost: costs,
                bounds: search_bounds(refs.search[l], (px + x) as i32, (py + y) as i32, w, h, vr),
            };
            s.run(cands, mc)
        };
    st.reset_mb();
    let pmv = [
        st.predict(0, 0, 0, 0, 16, 16),
        st.predict(1, 0, 0, 0, 16, 16),
    ];
    let mut m16 = [Mv::ZERO; 2];
    let mut c16 = [0u32; 2];
    for l in 0..2 {
        let mut cands = [Mv::ZERO; 8];
        let mut nc = 0;
        for &c in [pmv[l], dmv[0][l], Mv::ZERO]
            .iter()
            .chain(nb.cands[l][..nb.ncands[l]].iter())
        {
            if nc < cands.len() {
                cands[nc] = c;
                nc += 1;
            }
        }
        let (mv, _, c) = search(l, 0, 0, 16, 16, pmv[l], &cands[..nc], &cfg.me);
        m16[l] = mv;
        c16[l] = c;
    }
    let mvd16 = [sub(m16[0], pmv[0]), sub(m16[1], pmv[1])];
    // A pair, refined jointly: each list's vector a quarter sample
    // around, the other held, scored as the average they make.
    // `limit`: a pair whose start already costs more is left as it is.
    let bi_refine =
        |r: (usize, usize, usize, usize), m: [Mv; 2], p: [Mv; 2], limit: u32| -> ([Mv; 2], u32) {
            let mut mbi = m;
            let mut bi_cost =
                satd(r, PRED_BI, mbi) + costs.of(mbi[0], p[0]) + costs.of(mbi[1], p[1]);
            let rounds = if bi_cost > limit { 0 } else { cfg.bi_refine };
            for _ in 0..rounds {
                let before = mbi;
                for l in 0..2 {
                    let c = mbi[l];
                    for (dx, dy) in [
                        (0, -1),
                        (-1, 0),
                        (1, 0),
                        (0, 1),
                        (-1, -1),
                        (1, -1),
                        (-1, 1),
                        (1, 1),
                    ] {
                        let mut m = mbi;
                        m[l] = Mv::new(c.x + dx, c.y + dy);
                        let rc = costs.of(m[0], p[0]) + costs.of(m[1], p[1]);
                        if rc >= bi_cost {
                            continue;
                        }
                        let k = satd(r, PRED_BI, m) + rc;
                        if k < bi_cost {
                            bi_cost = k;
                            mbi = m;
                        }
                    }
                }
                if mbi == before {
                    break;
                }
            }
            (mbi, bi_cost)
        };
    let (mbi, bi_cost) = bi_refine((0, 0, 16, 16), m16, pmv, u32::MAX);
    let mvdbi = [sub(mbi[0], pmv[0]), sub(mbi[1], pmv[1])];
    let mut best16 = BCand::new(BMbKind::B16);
    let mut b16_est = f32::MAX;
    for dir in DIRS {
        let used = lists(dir);
        let est = if dir == PRED_BI {
            bi_cost as f32
        } else {
            let l = if used[0] { 0 } else { 1 };
            c16[l] as f32
        } + lam1 * b_type_bits(BMbKind::B16, (dir, dir));
        if est < b16_est {
            b16_est = est;
            best16 = BCand::new(BMbKind::B16);
            if dir == PRED_BI {
                best16.fill((0, 0, 16, 16), dir, mbi, mvdbi);
            } else {
                best16.fill((0, 0, 16, 16), dir, m16, mvd16);
            }
            best16.est = est;
        }
    }
    let mut shapes: [Option<BCand>; 4] = [Some(direct), Some(best16), None, None];

    if ctx.subparts {
        // B_8x8: per 8x8, direct or one direction at 8x8.
        st.reset_mb();
        let mut t = BCand::new(BMbKind::B8x8);
        t.est = lam1 * 6.0;
        for part in 0..4 {
            let r = ((part & 1) * 8, (part >> 1) * 8, 8, 8);
            let (x, y, w, h) = r;
            let mut bestp = (
                satd(r, ddir, dmv[part]) as f32 + lam1,
                SubMbShape::Direct,
                ddir,
                dmv[part],
                [Mv::ZERO; 2],
            );
            let p = [st.predict(0, 0, x, y, w, h), st.predict(1, 0, x, y, w, h)];
            let mut m = [Mv::ZERO; 2];
            let mut c = [0u32; 2];
            for l in 0..2 {
                let (mv, _, cc) = search(
                    l,
                    x,
                    y,
                    w,
                    h,
                    p[l],
                    &[m16[l], p[l], dmv[part][l]],
                    &cfg.me_sub,
                );
                m[l] = mv;
                c[l] = cc;
            }
            let (mb, cb) = bi_refine(r, m, p, (c[0].min(c[1]) as f32 * cfg.bi_part_ratio) as u32);
            for dir in DIRS {
                let est = match dir {
                    PRED_L0 => c[0] as f32,
                    PRED_L1 => c[1] as f32,
                    _ => cb as f32,
                } + lam1 * if dir == PRED_BI { 5.0 } else { 3.0 };
                if est < bestp.0 {
                    let mm = if dir == PRED_BI { mb } else { m };
                    bestp = (
                        est,
                        SubMbShape::S8x8,
                        dir,
                        mm,
                        [sub(mm[0], p[0]), sub(mm[1], p[1])],
                    );
                }
            }
            let (est, shape, dir, mv, mvd) = bestp;
            t.est += est;
            t.sub_shape[part] = shape;
            t.fill(r, dir, mv, mvd);
            let used = lists(dir);
            st.commit_part(
                x,
                y,
                w,
                h,
                [used[0].then_some(mv[0]), used[1].then_some(mv[1])],
            );
        }
        let c8 = t.est;
        // A B_8x8 of four direct sub-macroblocks is B_Direct_16x16 at a
        // higher price.
        if t.sub_shape.iter().any(|&s| s != SubMbShape::Direct) {
            shapes[2] = Some(t);
        }
        if c8 < b16_est {
            for kind in [BMbKind::B16x8, BMbKind::B8x16] {
                st.reset_mb();
                let mut c = BCand::new(kind);
                c.est = lam1 * 6.0;
                for &(x, y, w, h) in mb_partitions(kind.dec_kind()) {
                    let p = [st.predict(0, 0, x, y, w, h), st.predict(1, 0, x, y, w, h)];
                    let mut m = [Mv::ZERO; 2];
                    let mut cc = [0u32; 2];
                    for l in 0..2 {
                        let a = t.mv[(y / 4) * 4 + x / 4][l];
                        let b = t.mv[((y + h) / 4 - 1) * 4 + (x + w) / 4 - 1][l];
                        let (mv, _, k) =
                            search(l, x, y, w, h, p[l], &[a, b, m16[l], p[l]], &cfg.me_sub);
                        m[l] = mv;
                        cc[l] = k;
                    }
                    let mut bp = (f32::MAX, PRED_L0);
                    let (mb, cb) = bi_refine(
                        (x, y, w, h),
                        m,
                        p,
                        (cc[0].min(cc[1]) as f32 * cfg.bi_part_ratio) as u32,
                    );
                    for dir in DIRS {
                        let e = match dir {
                            PRED_L0 => cc[0] as f32,
                            PRED_L1 => cc[1] as f32,
                            _ => cb as f32,
                        };
                        if e < bp.0 {
                            bp = (e, dir);
                        }
                    }
                    let m = if bp.1 == PRED_BI { mb } else { m };
                    c.est += bp.0;
                    c.fill((x, y, w, h), bp.1, m, [sub(m[0], p[0]), sub(m[1], p[1])]);
                    let used = lists(bp.1);
                    st.commit_part(
                        x,
                        y,
                        w,
                        h,
                        [used[0].then_some(m[0]), used[1].then_some(m[1])],
                    );
                }
                let slot = if kind == BMbKind::B16x8 { 3 } else { 2 };
                if shapes[slot].is_none_or(|s| c.est < s.est) {
                    shapes[slot] = Some(c);
                }
            }
        }
    }
    let best_est = shapes
        .iter()
        .flatten()
        .map(|c| c.est)
        .fold(f32::MAX, f32::min);
    let i_est = intra_estimate(ctx, rec, mb_x, mb_y, src, nb.avail);

    if !cfg.rd {
        let best = *shapes
            .iter()
            .flatten()
            .min_by(|a, b| a.est.total_cmp(&b.est))
            .expect("direct is always a candidate");
        if i_est < best.est * cfg.intra_ratio {
            let (idec, modes, e) = intra_mb(
                &cfg,
                ctx,
                rec,
                mb_x,
                mb_y,
                src,
                nb,
                qp_pred,
                None,
                &mut |_: &MbDecision| 0.0,
            );
            if (e as f32) < best.est {
                st.reset_mb();
                return MbChoice::Intra(idec, modes);
            }
        }
        predict_cand(rec, st, &best);
        let d0 = best.decision();
        let allow8 = ctx.t8x8 && d0.no_sub_mb_part_less_than_8x8();
        let r = code_inter_mb_residual(
            ctx,
            rec,
            mb_x,
            mb_y,
            src.y,
            src.ys,
            src.c,
            src.cs,
            allow8,
            Some(allow8),
        );
        let residual = r.cbp_luma != 0 || r.cbp_chroma != 0;
        let kind = if best.kind == BMbKind::BDirect16 && !residual {
            BMbKind::BSkip
        } else {
            best.kind
        };
        return MbChoice::Inter(BDecision {
            kind,
            transform_8x8: r.transform_8x8,
            cbp_luma: r.cbp_luma,
            cbp_chroma: r.cbp_chroma,
            qp_delta: delta(ctx.qp, qp_pred, residual),
            luma: r.luma,
            chroma_dc: r.chroma_dc,
            chroma_ac: r.chroma_ac,
            nz_luma: r.nz_luma,
            nz_chroma: r.nz_chroma,
            ..d0
        });
    }

    let rctx_store;
    let rctx: &IntraCtx<S> = match an.trellis().filter(|_| cfg.trellis_rd && !ctx.c444) {
        Some(t) => {
            rctx_store = IntraCtx {
                trellis: Some(t),
                ..*ctx
            };
            &rctx_store
        }
        None => ctx,
    };
    let mut best_j = skip_ssd as f64 + lam2 * rate(BMb::Skip(&skip_dec));
    let mut best: MbChoice<BDecision> = MbChoice::Inter(skip_dec);
    let mut best_pix = MbPix::new();
    let mut best_st = direct_st;
    let mut skip_won = true;
    let mut pred_pix = MbPix::new();
    let mut best_pred = MbPix::new();
    let mut best_t8 = (false, false);
    let mut order: Vec<&BCand> = shapes.iter().flatten().collect();
    order.sort_by(|a, b| a.est.total_cmp(&b.est));
    for cand in order.into_iter().take(cfg.rd_max) {
        if cand.est > best_est * cfg.rd_ratio {
            continue;
        }
        predict_cand(rec, st, cand);
        let d0 = cand.decision();
        let allow8 = ctx.t8x8 && d0.no_sub_mb_part_less_than_8x8();
        let tries: &[bool] = if allow8 { &[false, true] } else { &[false] };
        pred_pix.save(ctx, rec, mb_x, mb_y);
        let mut first: Option<([[i16; 16]; 2], [[[i16; 16]; 16]; 2], [[u8; 16]; 2], u8)> = None;
        for (k, &t8) in tries.iter().enumerate() {
            if k > 0 {
                if ctx.c444 {
                    pred_pix.restore(ctx, rec, mb_x, mb_y);
                } else if first.is_none() {
                    break;
                } else {
                    pred_pix.restore_luma(rec, mb_x, mb_y);
                }
            }
            let mut r = code_inter_mb_residual_parts(
                rctx,
                rec,
                mb_x,
                mb_y,
                src.y,
                src.ys,
                src.c,
                src.cs,
                allow8,
                Some(t8),
                k == 0 || ctx.c444,
            );
            if k > 0 && !ctx.c444 {
                if let Some((dc, ac, nz, cbp)) = first {
                    (r.chroma_dc, r.chroma_ac, r.nz_chroma, r.cbp_chroma) = (dc, ac, nz, cbp);
                }
            } else if k == 0 {
                first = Some((r.chroma_dc, r.chroma_ac, r.nz_chroma, r.cbp_chroma));
                if allow8 && r.cbp_luma == 0 && !ctx.c444 {
                    first = None;
                }
            }
            if t8 && r.cbp_luma == 0 {
                continue;
            }
            let residual = r.cbp_luma != 0 || r.cbp_chroma != 0;
            if cand.kind == BMbKind::BDirect16 && !residual {
                continue; // B_Skip, priced above
            }
            let dec = BDecision {
                transform_8x8: r.transform_8x8,
                cbp_luma: r.cbp_luma,
                cbp_chroma: r.cbp_chroma,
                qp_delta: delta(ctx.qp, qp_pred, residual),
                luma: r.luma,
                chroma_dc: r.chroma_dc,
                chroma_ac: r.chroma_ac,
                nz_luma: r.nz_luma,
                nz_chroma: r.nz_chroma,
                ..d0.clone()
            };
            let mb = if cand.kind == BMbKind::BDirect16 {
                BMb::Direct(&dec)
            } else {
                BMb::Explicit(&dec)
            };
            let j = ssd_mb(ctx, rec, src, mb_x, mb_y) as f64 + lam2 * rate(mb);
            if j < best_j {
                best_j = j;
                best = MbChoice::Inter(dec);
                best_pix.save(ctx, rec, mb_x, mb_y);
                best_pred.clone_from(&pred_pix);
                best_t8 = (allow8, t8);
                best_st = *st;
                skip_won = false;
            }
        }
    }

    if i_est < best_est * cfg.intra_ratio {
        let (idec, modes, j) = intra_mb(
            &cfg,
            rctx,
            rec,
            mb_x,
            mb_y,
            src,
            nb,
            qp_pred,
            None,
            &mut |d: &MbDecision| rate(BMb::Intra(d)),
        );
        let j =
            j + (ssd_mb(ctx, rec, src, mb_x, mb_y) - ssd_luma(ctx, rec, src, mb_x, mb_y)) as f64;
        if j < best_j {
            st.reset_mb();
            let mut idec = idec;
            if let Some(t) = an.trellis().filter(|_| !ctx.c444 && !cfg.trellis_rd) {
                let tctx = IntraCtx {
                    trellis: Some(t),
                    ..*ctx
                };
                requant_intra(&tctx, rec, mb_x, mb_y, src, nb, &mut idec, &modes, qp_pred);
            }
            return MbChoice::Intra(idec, modes);
        }
    }
    *st = best_st;
    if skip_won {
        predict_cand(rec, st, &direct);
        return best;
    }
    if let (Some(t), MbChoice::Inter(dec)) = (
        an.trellis().filter(|_| !ctx.c444 && !cfg.trellis_rd),
        &mut best,
    ) {
        best_pred.restore(ctx, rec, mb_x, mb_y);
        let tctx = IntraCtx {
            trellis: Some(t),
            ..*ctx
        };
        let r = code_inter_mb_residual(
            &tctx,
            rec,
            mb_x,
            mb_y,
            src.y,
            src.ys,
            src.c,
            src.cs,
            best_t8.0,
            Some(best_t8.1),
        );
        let residual = r.cbp_luma != 0 || r.cbp_chroma != 0;
        dec.transform_8x8 = r.transform_8x8;
        dec.cbp_luma = r.cbp_luma;
        dec.cbp_chroma = r.cbp_chroma;
        dec.luma = r.luma;
        dec.nz_luma = r.nz_luma;
        dec.chroma_dc = r.chroma_dc;
        dec.chroma_ac = r.chroma_ac;
        dec.nz_chroma = r.nz_chroma;
        dec.qp_delta = delta(ctx.qp, qp_pred, residual);
        if dec.kind == BMbKind::BDirect16 && !residual {
            dec.kind = BMbKind::BSkip;
        }
    } else {
        best_pix.restore(ctx, rec, mb_x, mb_y);
    }
    best
}
