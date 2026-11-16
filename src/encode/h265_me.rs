//! Motion estimation and inter mode decision for H.265 P coding units.
//!
//! The inter counterpart of [`super::h265_intra`], and the H.265 sibling of
//! [`super::h264_me`]: the *deciding* half of coding a P CU — which motion,
//! through which signalling (skip, merge, or AMVP plus a difference), what
//! the quantised coefficients are, and what the reconstruction looks like.
//! Turning that into bits belongs to the CABAC coding-tree writer; the seam
//! is [`InterCuDecision`] — data, not a shared file, shaped after
//! [`super::h265_intra::CuDecision`] so the serialisers meet a familiar
//! layout.
//!
//! The two rules of the other decision modules hold unchanged:
//!
//! **The prediction is the decoder's own.** The chosen candidate's
//! prediction runs through `crate::hevc::inter::predict_block` — the very
//! function the decoder's `prediction_unit` calls, fused kernels, window
//! clamping and all — reading the *reconstructed* reference frame. An
//! encoder that predicted from source samples, or interpolated one rounding
//! differently, would desync, and inter desync compounds picture over
//! picture.
//!
//! **So is the signalling derivation — not mirrored, called.** Choosing a
//! `merge_idx` or an `mvp_l0_flag` requires knowing the candidate lists the
//! decoder will build, and this module gets them by calling the decoder's
//! own `crate::hevc::mvpred::merge_candidate` and
//! `crate::hevc::mvpred::amvp` — the functions `prediction_unit` in
//! `src/hevc/ctu.rs` calls, conformance-proven against the JCT-VC suites.
//! Drift between the two derivations is therefore impossible rather than
//! merely tested against. What that costs is state: those functions read a
//! decoder-grade `PicInfo` (z-scan availability, 6.4.2) and the frame's
//! per-4x4 motion grid, so [`InterPicture`] maintains both exactly as the
//! decoder does — the motion fill mirrors the "Store motion" block of
//! `prediction_unit` (`ref_delta` as POC differences, long-term flags), and
//! the CTB / pred-mode marks mirror what `available_at` consults. The
//! replay test at the bottom re-derives every decision on an independently
//! maintained state to prove the maintenance, not the derivation.
//!
//! **And so is chroma, in every format.** All four chroma formats are
//! coded, and none of the three derivations that differ between them is
//! written out here:
//!
//! - The **chroma motion vector** comes from `predict_block` itself, which
//!   reads `(SubWidthC, SubHeightC)` off the frame it is writing into and
//!   scales by `mv * 2 / SubWidthC` per axis (its `mvc` closure,
//!   8.5.3.2.10). Building [`InterPicture::recon`] in the SPS's own format
//!   is therefore the entire chroma-MC change: 4:2:2's horizontally
//!   unscaled, vertically doubled vector is the decoder's derivation, not
//!   a second copy of it. This is the one rule self-consistency cannot
//!   check — an encoder and a decoder sharing a wrong vector agree with
//!   each other — so it is CROSS against HM that arbitrates it.
//! - The **chroma transform blocks** are placed by
//!   `h265_intra::chroma_tbs`, which is `transform_unit`'s `here`
//!   expression plus its `yct = yc + t * nc` stacked-pair loop.
//! - The **chroma QP** is `hevc::ctu::chroma_qp` told this stream's
//!   `ChromaArrayType`: Table 8-10 for 4:2:0 and `Min(qPi, 51)` for
//!   everything else. Note for anyone testing this: the two agree for
//!   every `qPi` below 30, so a fixed-QP-26 stream cannot tell them apart.
//!
//! **Temporal MVP follows the tier.** Under `h265_tune::Tune::tmvp` the
//! SPS enables it and every inter slice sets
//! `slice_temporal_mvp_enabled_flag`, naming `RefPicList1[0]` (a B
//! picture's future anchor) or a P picture's `RefPicList0[0]` as the
//! collocated picture; [`InterPicture::tmvp`] puts the same picture in the
//! `RefCtx`, so the decoder's derivation adds the temporal candidate to
//! both lists here exactly as it will there. Without it the SPS clears
//! the flag and the spatial + zero candidate set is the *complete*
//! derivation. The collocated picture's motion grid is the one this
//! module filled when it was coded (an intra picture's is marked intra
//! when it is kept, `Core::retain_reference`).
//!
//! # Prediction-unit partitions (2026-09-18)
//!
//! Under `Config::inter_parts` a CU may also take the symmetric halves,
//! `PART_2NxN` and `PART_Nx2N` ([`InterCuKind::Parts`]), at every CU size,
//! in P and B pictures. Each shape's two units are decided in the reader's
//! order by the same pickers as a 2Nx2N unit (`InterPicture::pick_pu`,
//! `InterPicture::pick_pu_b`), each over the decoder's own candidates
//! for its `PuPos` — the second unit's derived with the first unit's
//! motion already in the motion grid, which is what the second-unit
//! exclusions of the merge derivation depend on. An 8x4 or 4x8 unit never
//! takes AMVP `PRED_BI` (the reader has no spelling for it) and its merge
//! candidates are made one-list by the derivation itself. The residual
//! stays one CU-sized TU: this SPS's `max_transform_hierarchy_depth_inter`
//! is 2, so no transform split is inferred (`interSplitFlag` is 0). A
//! partitioned CU codes `rqt_root_cbf`, which the reader infers only for a
//! 2Nx2N merge, and is never a skip.
//!
//! The cheapest shape in SATD plus bits (the units' own syntax, then
//! `part_mode` and the CU syntax around them) meets the 2Nx2N unit, and
//! when the two come within `PARTS_RD_MARGIN` of each other both are
//! coded and the cheaper in SSD plus lambda times the CU's bits is kept.
//! That choice is the CU's unsplit candidate in the coding quadtree, whose
//! split decision is unchanged. The measurements are on
//! `PARTS_RD_MARGIN` and `Config::inter_parts`.
//!
//! What was priced first (2026-09-13), for the record: measured while the
//! encoder coded one `PART_2Nx2N` unit per CTB,
//! `tools/partition_opportunity.py` (integer-sample SAD, ±4, a split must
//! beat the whole block by 5%) over the seven 8-bit 4:2:0 clips found
//! 16.2% of blocks would take a symmetric split and 13.3% an AMP shape
//! beyond it — 7.1% symmetric once the smooth-gradient clip is set aside,
//! the AMP count mostly the fractal half of the cut clip. The transform
//! split it expected to cost was a misreading of this SPS, whose inter
//! transform depth was already 2.
//!
//! # Priced, not built (2026-09-13)
//!
//! Two of the standard's inter tools were costed against this corpus
//! before deciding not to build them. The numbers are here so the next
//! reader decides against a bigger corpus rather than re-deriving them.
//!
//! - **Temporal merge / AMVP candidates (TMVP).** Costed here at under
//!   half a percent from the AMVP share of a QP-26 census, and built
//!   anyway on 2026-11-15 when x265's own ablation (black box) put it at
//!   1-3%: it measured -1.1% BD-rate at medium against x265 (live
//!   contexts already on), most on mobile (-1.9%) and park_joy (-0.8%).
//!   The census undercounted it: the temporal candidate also turns merges
//!   and skips that were second-best into the cheapest choice.
//! - **WPP / tile-parallel coding.** A speed tool, not a compression
//!   one, and the corpus cannot measure it: every clip is two CTB rows.
//!   The ceiling is rows over two (entropy sync lets a row start when
//!   the row above is two CTBs ahead) — nothing at 64x64, up to ~17x on
//!   1080p at CTB 32 in the limit — bought with `entry_point_offset`s in
//!   the slice header, a context save after the second CTB of each row,
//!   and a decision walk that runs per row behind the same lag. rivet
//!   already parallelises across pictures and chunks, so this would buy
//!   latency rather than throughput there; the timing that would justify
//!   it needs a clip with seconds and rows, which the corpus lacks.
//!
//! # Scope (v1) — the same deliberately fixed geometry as the intra module
//!
//! - **P slices, one reference** (list 0, `ref_idx` 0), `PART_2Nx2N` CUs
//!   (and, under `Config::inter_parts`, two-unit ones: see above)
//!   — a whole CTB, or a quadtree leaf down to 8x8 through
//!   `InterPicture::code_ctu_tree` — each with one CU-sized TU (no transform
//!   split — the SPS's
//!   maximum transform size equals the CTB size precisely so this shape is
//!   representable). The quantiser and the weighting are the caller's:
//!   per-unit quantisers through `MeCtx::qp`, and list 0's explicit
//!   weighting through [`InterPicture::wp`] when the slice carries a
//!   `pred_weight_table` (`Weighting::Default` otherwise), and a B
//!   slice's list-0, list-1 and bi weighting through
//!   [`InterPicture::wp_b`].
//! - **All four chroma formats.** Monochrome omits every chroma element,
//!   mirroring the reader's uniform `chroma_array_type != 0` gates in
//!   `transform_tree` and `transform_unit`; 4:2:0 carries one half-size
//!   chroma TB per component; 4:2:2 the stacked pair of half-size squares,
//!   each with its own cbf; 4:4:4 one chroma TB at the luma TB's own size.
//!   The unsplit depth-0 node is above 4x4 in every geometry this module
//!   produces, so `transform_unit`'s chroma-at-the-parent case (its
//!   `blk_idx == 3` arm, for 4x4 luma TBs) never arises here.
//! - **`MaxNumMergeCand` = 5**: the slice header this decision assumes
//!   writes `five_minus_max_num_merge_cand` = 0. The PPS writes
//!   `log2_parallel_merge_level_minus2` = 0 (level 2: no merge-list
//!   sharing), and `RefCtx` says both.
//!
//! # The search, named
//!
//! Full-sample: a greedy small-diamond descent on SAD over the reference
//! luma plane, seeded at the two AMVP predictors, the zero vector and every
//! merge candidate, confined to ±`SEARCH_RANGE` full samples around the
//! best seed and to the padded plane (HEVC clamps out-of-picture reads, so
//! any vector is *legal* — the confinement merely keeps the SAD reads
//! direct). Then two SATD refinement rings — the eight half-sample
//! neighbours, then the eight quarter-sample neighbours of the winner —
//! scored on a luma-only prediction through the decoder's own kernels
//! (the addressing mirrors `interp` / `source` in `src/hevc/inter.rs`).
//! Merge candidates are scored at their exact vectors. The one rate
//! heuristic is `lambda` times an approximate bin count per signalling
//! shape — the same single-function placeholder policy as the intra
//! module's `mode_signalling_cost`, replaced wholesale when a real bit
//! count exists.
//!
//! # What the writer serialises (the seam contract)
//!
//! For a P-slice 2Nx2N inter CU, in the reader's order (`coding_unit` /
//! `prediction_unit` in `src/hevc/ctu.rs`):
//!
//! - [`InterCuKind::Skip`]: `cu_skip_flag` 1, `merge_idx` (TR, cMax
//!   `MaxNumMergeCand` − 1). Nothing else — `rqt_root_cbf` is inferred 0.
//! - [`InterCuKind::Merge`]: `cu_skip_flag` 0, `pred_mode_flag` 0,
//!   `part_mode` 2Nx2N, `merge_flag` 1, `merge_idx`. `rqt_root_cbf` is
//!   **not coded** — the reader infers it true for a non-skip 2Nx2N merge
//!   CU (ctu.rs's `!(part_mode == P2Nx2N && last_pu_merged)` gate), which
//!   is why this module only produces `Merge` when residual survived (a
//!   zero-residual merge becomes `Skip`; an invariant a test holds).
//!   Then the transform tree, whose shape is the format's (below).
//! - [`InterCuKind::Amvp`]: `cu_skip_flag` 0, `pred_mode_flag` 0,
//!   `part_mode` 2Nx2N, `merge_flag` 0; no `inter_pred_idc` (P slice), no
//!   `ref_idx_l0` (one active reference); `mvd_l0` (as
//!   `abs_mvd_greater0_flag`, `abs_mvd_greater1_flag`, `abs_mvd_minus2`,
//!   `mvd_sign_flag` per component), `mvp_l0_flag`; then `rqt_root_cbf`,
//!   and when set the same unsplit transform tree.
//! - [`InterCuKind::Parts`]: `cu_skip_flag` 0, `pred_mode_flag` 0,
//!   `part_mode` (2NxN `01`, Nx2N `00`, context-coded, the inverse of
//!   `parse_part_mode`'s inter arms: `hevc::ctu::write_part_mode_inter_at`),
//!   then each unit's `prediction_unit` — `merge_flag`, then `merge_idx` or
//!   the AMVP syntax, `inter_pred_idc` at the unit's own dimensions in a B
//!   slice (`write_prediction_unit`) — then `rqt_root_cbf`, coded, and
//!   when set the same unsplit transform tree.
//! - [`InterCuKind::UseIntra`]: not an inter CU at all. This decision's
//!   coefficients are meaningless and the planes are untouched; the
//!   caller calls [`InterPicture::code_ctu_intra`], which runs the intra
//!   decision over this same picture's reconstruction, and serialises
//!   the [`CuDecision`] that returns instead. See [`prefer_intra`], and
//!   [`PCuDecision`] for the seam that carries either kind.
//!
//! The transform tree, at `split_transform_flag` 0 and depth 0, spells in
//! the reader's order (`transform_tree`, then `transform_unit`):
//!
//! 1. `split_transform_flag`, 0.
//! 2. The chroma cbfs, **only when `chroma_array_type != 0`**: per
//!    component `cbf_c[c][0]`, and at 4:2:2 `cbf_c[c][1]` immediately
//!    after it (`transform_tree`'s `cat == 2 && (!split || log2 == 3)`
//!    arm — an unsplit node always codes both halves of the pair). The
//!    fields are [`InterCuDecision::cbf_chroma`] and
//!    [`InterCuDecision::cbf_chroma_bot`].
//! 3. `cbf_luma` — **but only if some chroma cbf above was set.** At an
//!    inter leaf of depth 0 with every chroma cbf clear the reader codes
//!    no bin and infers `cbf_luma` 1, so writing one desyncs. Monochrome
//!    has no chroma cbf to set and therefore never carries the bin at
//!    all; such a CU must genuinely have luma coefficients, which this
//!    module guarantees by spelling a residual-free CU as a skip or as
//!    `rqt_root_cbf` 0.
//! 4. The luma residual, if `cbf_luma`.
//! 5. The chroma residuals, components outermost and the 4:2:2 pair
//!    within (`transform_unit`'s `for c` around its `for t`), each at the
//!    TB size and slot [`InterCuDecision::chroma`] documents.
//!
//! Coefficients are raster within each TB, exactly as in `CuDecision`; the
//! writer derives the scan (always diagonal for inter TBs — the
//! mode-dependent scans are intra-only, and `residual_scan_idx` returns 0
//! for every non-intra block, 7.4.9.11).
//!
//! # Duplication, flagged
//!
//! `code_residual_inter` is `h265_intra::code_residual` minus the intra
//! DST case and with the inter quantisation offset — copied, not shared,
//! because sharing would mean editing that module's private function into a
//! public one under a parallel delivery. Fold the two together when the
//! files next change hands. `lambda` duplicates the intra module's
//! Lagrangian constant for the same reason.

use crate::cabac_enc::CabacEncoder;
use crate::dsp::hevc_enc::{qbits, quant_offset, quant_scale};
use crate::encode::InterParts;
use crate::encode::h265_intra::{
    CuDecision, Geo, IntraCtx, MIN_CB_LOG2, RegionSave, Srcs, TreeCu, chroma_tbs,
    code_cu_2nx2n_intra, code_cu_nxn_intra, cu_ssd, restore4, satd_lambda_scale, save4, save4_into,
    ssd_lambda, sub_wh,
};
use crate::encode::h265_search;
use crate::hevc::ctu::{
    PartMode, SplitCuNb, chroma_qp, explicit_weighting, write_cu_skip_flag, write_inter_pred_idc,
    write_merge_flag, write_merge_idx, write_mvd, write_mvp_flag, write_part_mode_inter,
    write_part_mode_inter_at, write_pred_mode_flag, write_ref_idx, write_rqt_root_cbf,
    write_split_cu_flag,
};
use crate::hevc::ctx::Contexts;
use crate::hevc::frame::{Frame, MotionInfo, Mv, Plane16, fill_motion};
use crate::hevc::inter::{McScratch, Weighting, predict_block};
use crate::hevc::intra::IntraScratch;
use crate::hevc::mvpred::{Cand, PuPos, RefCtx, amvp, merge_candidates};
use crate::hevc::pic::{Geometry, PicInfo};
use crate::hevc::pps::Pps;
use crate::hevc::residual::{ScalingSource, scale_coefficients};
use crate::hevc::slice::PredWeightTable;
use crate::hevc::sps::Sps;
use crate::sample::Sample;

/// Everything inter coding needs that does not change per CU — the *same*
/// struct the intra module takes, aliased rather than repeated, for the
/// H.264 side's reason: one context serving both modules means the picture
/// loop cannot hand the two halves of a CU decision different QPs.
pub type MeCtx<'a, S> = IntraCtx<'a, S>;

/// `MaxNumMergeCand` this decision assumes the slice header will declare
/// (`five_minus_max_num_merge_cand` = 0).
pub const MAX_MERGE_CAND: usize = 5;

/// Full-sample search confinement around the best seed, in luma samples.
/// Kept well inside the frame's `LUMA_PAD` (80) so every full-sample SAD
/// reads the padded plane directly.
const SEARCH_RANGE: i32 = 48;

/// How close, in SATD plus bits, the losing candidate of a CU's 2Nx2N unit
/// and its best partitioned shape must come to the winner for `tree_leaf`
/// to code it too and choose between the two in SSD plus lambda times the
/// CU's bits: within this fraction of the winner's cost (1.0: at most
/// twice it).
///
/// The SATD comparison alone misjudges the choice, because it knows
/// neither the residual each candidate leaves nor that a 2Nx2N merge with
/// nothing left codes as a skip. Measured 2026-09-18, `--parts sym`
/// against `--parts none` on one binary, YUV BD-rate at QP 22-40 (the
/// 1280x720 clips at 22-37), IP / IPB:
///
/// ```text
///   margin              corpus mean (11 clips)   720p testsrc2    720p natural
///   SATD only           -0.93% / -0.91%          -1.71 / -1.06    -0.10 / -0.18
///   0.1                 -1.43% / -1.37%
///   0.3                 -1.58% / -1.67%          -2.47 / -2.44    -1.40 / -1.38
///   1.0                 -1.59% / -1.91%          -2.69 / -2.38    -1.54 / -1.33
///   any (always)                                 -2.64 / -2.35    -1.31 / -1.32
/// ```
///
/// On the natural clip SATD alone let B pictures take shapes that cost
/// more than they saved (its B slices grew 0.6%). The second coding costs
/// little: CPU against `none` went from x1.50-2.00 with SATD alone to
/// x1.62-2.18 at 1.0.
const PARTS_RD_MARGIN: f32 = 1.0;

/// The CU-level choices this module decides between.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum InterCuKind {
    /// `cu_skip_flag` 1: merge motion, no residual syntax at all. Chosen
    /// exactly when a merge candidate wins the cost comparison *and* no
    /// level survived quantisation.
    Skip {
        /// `merge_idx`.
        merge_idx: u8,
    },
    /// Non-skip 2Nx2N merge: merge motion with a coded residual
    /// (`rqt_root_cbf` inferred 1 by the reader — see the module header).
    Merge {
        /// `merge_idx`.
        merge_idx: u8,
    },
    /// AMVP: `mvp_l0_flag` picks the predictor, `mvd_l0` carries the rest.
    Amvp {
        /// `mvp_l0_flag`.
        mvp_flag: u8,
        /// `mv` minus the chosen predictor, wrapping i16 (the reader's
        /// `uLX` sum is a wrapping add, ctu.rs).
        mvd: Mv,
    },
    /// B: AMVP in one or both lists. `idc` is `inter_pred_idc` in the
    /// reader's own encoding — 0 `PRED_L0`, 1 `PRED_L1`, 2 `PRED_BI` — and
    /// only the entries of `mvd` / `mvp_flag` whose list the `idc` uses
    /// are meaningful; the other is zero and must not be written.
    ///
    /// What the writer spells, in `prediction_unit`'s order: `merge_flag`
    /// 0, then `inter_pred_idc` through
    /// `hevc::ctu::write_inter_pred_idc` (whose docblock carries the
    /// `w + h != 12` reading), then **per list, interleaved** — L0's
    /// `ref_idx` / `mvd` / `mvp_flag`, then L1's, not both `ref_idx`
    /// followed by both `mvd`. `ref_idx` is absent in both lists because
    /// each declares exactly one active reference.
    BAmvp {
        /// `inter_pred_idc`: 0 L0, 1 L1, 2 BI.
        idc: u8,
        /// `mvd_l0` / `mvd_l1`, wrapping i16 differences from the chosen
        /// predictor of that list.
        mvd: [Mv; 2],
        /// `mvp_l0_flag` / `mvp_l1_flag`.
        mvp_flag: [u8; 2],
    },
    /// Two prediction units ([`InterCuDecision::part`] says how the CU is
    /// divided, [`InterCuDecision::pus`] what each carries), each merge or
    /// AMVP on its own. `rqt_root_cbf` is coded: the reader infers it only
    /// for a 2Nx2N merge unit, so a partitioned CU whose residual quantised
    /// away codes a 0 rather than becoming a skip, which is 2Nx2N only.
    Parts,
    /// Inter lost to the flatness proxy: code this CU with the intra
    /// decision instead. Only `mv` is meaningful (callers feeding motion
    /// state must *not* use it — an intra CU's motion is
    /// `MotionInfo::INTRA`, which `InterPicture::code_ctu` has already
    /// stored in that case).
    UseIntra,
}

/// One prediction unit's motion decision: which motion it carries, how it
/// is signalled and what that costs. What `InterPicture::pick_pu` and
/// `InterPicture::pick_pu_b` return, and, for the two units of a
/// [`InterCuKind::Parts`] CU, what the writer spells.
#[derive(Clone, Copy, Debug)]
pub struct PuPick {
    /// `merge_idx` when a merge candidate won, `None` for AMVP.
    pub merge_idx: Option<u8>,
    /// The AMVP shape's `inter_pred_idc` (0 `PRED_L0`, 1 `PRED_L1`,
    /// 2 `PRED_BI`); 0 in a P slice.
    pub idc: u8,
    /// AMVP's `mvd` per list, wrapping differences from the chosen
    /// predictors. Only the lists `idc` uses are written.
    pub mvd: [Mv; 2],
    /// AMVP's `mvp_lX_flag` per list.
    pub mvp_flag: [u8; 2],
    /// The motion as the decoder stores it: a vector and a reference index
    /// per list, -1 for a list the unit does not use.
    pub mv: [Mv; 2],
    /// See `mv`.
    pub ref_idx: [i8; 2],
    /// Luma SATD of the winner's prediction.
    pub satd: u32,
    /// The winner's SATD plus lambda times the bits it was priced at: a
    /// whole 2Nx2N CU's own syntax for the CU's single unit, the unit's
    /// own syntax for one of two.
    pub cost: f32,
}

/// A partitioned shape's decision, [`InterPicture::pick_parts`]'s answer:
/// the shape, its two units in the reader's order, and its cost in SATD
/// plus bits, comparable with a 2Nx2N unit's [`PuPick::cost`].
type PartsPick = (PartMode, [PuPick; 2], f32);

impl Default for PuPick {
    fn default() -> Self {
        PuPick {
            merge_idx: None,
            idc: 0,
            mvd: [Mv::ZERO; 2],
            mvp_flag: [0; 2],
            mv: [Mv::ZERO; 2],
            ref_idx: [0, -1],
            satd: 0,
            cost: 0.0,
        }
    }
}

/// How one P CU was coded, in the form the coding-tree writer needs.
/// Produced once per CTU and meant to be consumed immediately.
#[derive(Clone)]
pub struct InterCuDecision {
    /// log2 of the CU (== CTB) size, 4 or 5.
    pub log2_cu: u32,
    /// `cu_transquant_bypass_flag`. When set, `luma` and `chroma` carry
    /// the **raw residual** rather than quantised levels: the decoder
    /// skips dequantisation and the inverse transform for such a CU and
    /// adds what it reads straight to the prediction, so prediction plus
    /// residual is the source exactly.
    ///
    /// Two consequences the writer and the deblocker must respect.
    /// `coding_unit` reads this flag as the CU's **very first** bin —
    /// before `cu_skip_flag`, so even a skipped CU spells it — and only
    /// when the PPS sets `transquant_bypass_enabled_flag`. And a bypass
    /// CU is exempt from the in-loop filters sample for sample, which is
    /// what keeps a lossless stream lossless once deblocking is on.
    pub bypass: bool,
    /// Whether the levels were quantised for a PPS that hides signs: see
    /// `CuDecision::sign_hiding`.
    pub sign_hiding: bool,
    /// `QpY` of this CU as a decoder will hold it: the context quantiser
    /// when the CU carries residual, the *predicted* one when it does not
    /// — a skip or a root-cbf-0 CU codes no `cu_qp_delta`. Filled with
    /// the context quantiser here; the encoder's quantiser chain
    /// overwrites it when the picture varies the quantiser per CTB. Read
    /// by the deblocker. See `CuDecision::qp_y`.
    pub qp_y: i32,
    /// The choice, with its signalling payload.
    pub kind: InterCuKind,
    /// `part_mode`: `PART_2Nx2N` for every kind but [`InterCuKind::Parts`].
    pub part: PartMode,
    /// A [`InterCuKind::Parts`] CU's two prediction units, in the reader's
    /// order (`PartMode::pus`). Unused otherwise; the single unit of the
    /// other kinds is `kind` plus the motion fields below, which for a
    /// partitioned CU repeat the first unit's.
    pub pus: [PuPick; 2],
    /// The chosen motion vector, quarter luma samples, list 0. Filled for
    /// every kind including `Skip` (the writer carries no vector, but
    /// callers and tests want the motion the CU actually has).
    pub mv: Mv,
    /// Reference index into list 0, or -1 when this CU does not use
    /// list 0 (only a B CU can say that). Otherwise always 0: this module
    /// searches exactly one reference per list. The field exists so the
    /// serialiser's contract does not change when more arrive.
    pub ref_idx: i8,
    /// The list-1 motion vector, quarter luma samples. Meaningful only
    /// when [`InterCuDecision::ref_idx_l1`] is not -1, which only a B CU
    /// can arrange; a P decision leaves this zero.
    pub mv_l1: Mv,
    /// Reference index into list 1, -1 when the CU does not use list 1.
    /// Always -1 for P (a P slice has no list 1 at all) and for a B CU
    /// whose `inter_pred_idc` is `PRED_L0`.
    pub ref_idx_l1: i8,
    /// `rqt_root_cbf` as the *reader* resolves it: false for `Skip`, true
    /// for `Merge` (inferred, never coded), the explicit value for `Amvp`
    /// (coded). When false, no transform tree follows and every
    /// coefficient below is zero.
    pub rqt_root_cbf: bool,
    /// `cbf_luma` of the single CU-sized luma TU.
    pub cbf_luma: bool,
    /// `cbf_cb` / `cbf_cr` of this CU's *first* (or only) chroma TB per
    /// component — the `cbf_c[c][0]` bin `transform_tree` codes. At 4:2:2
    /// this is the top square of the stacked pair; see
    /// [`InterCuDecision::cbf_chroma_bot`] for the bottom one. Always
    /// false when `chroma_array_type` is 0, where no chroma bin exists.
    pub cbf_chroma: [bool; 2],
    /// 4:2:2 only: `cbf_c[c][1]`, the stacked pair's *bottom* square.
    /// `transform_tree` codes it right after `cbf_c[c][0]` on an unsplit
    /// node (its `cat == 2 && (!split || log2 == 3)` arm), so the writer
    /// spells the two bins adjacently, per component. False in every other
    /// format. Named as in [`super::h265_intra::CuDecision`], whose
    /// serialiser has the same shape.
    pub cbf_chroma_bot: [bool; 2],
    /// Quantised luma levels of the one `n x n` TU (`n = 1 << log2_cu`),
    /// raster, at `[0..n*n]`. Entries beyond are zero and meaningless.
    pub luma: [i16; 1024],
    /// Quantised chroma levels per component (`[0]` Cb, `[1]` Cr),
    /// raster within each chroma TB, packed one TB per slot of
    /// `nc * nc` where `nc = 1 << log2c` — the chroma TB edge
    /// `transform_unit` derives (`here`'s `if cat == 3 { log2 } else
    /// { log2 - 1 }`). Slot `t` occupies `[t*nc*nc .. (t+1)*nc*nc]`, in
    /// the reader's own coding order:
    ///
    /// | `chroma_array_type` | `log2c` | slots | where each TB sits |
    /// |---|---|---|---|
    /// | 0 (4:0:0) | — | none | no chroma syntax exists at all |
    /// | 1 (4:2:0) | `log2_cu - 1` | 1 | `(x0/2, y0/2)`, one half-size square |
    /// | 2 (4:2:2) | `log2_cu - 1` | 2 | `(x0/2, y0)` then `(x0/2, y0 + nc)` — the stacked pair, top slot 0, bottom slot 1 |
    /// | 3 (4:4:4) | `log2_cu` | 1 | `(x0, y0)`, at the luma TB's own size |
    ///
    /// Positions are chroma-plane coordinates; the placement is
    /// `h265_intra::chroma_tbs`, which mirrors `transform_unit`'s
    /// `here` and its `yct = yc + t * nc` loop. Entries beyond the slots a
    /// format uses are zero and meaningless. Sized for the largest case,
    /// a 4:4:4 32x32 chroma TB.
    pub chroma: [[i16; 1024]; 2],
    /// The transform tree splits once: four quarter-size luma TUs in
    /// z-order, `luma` packing quadrant `i` at `[i * q .. (i + 1) * q]`
    /// (`q = (n / 2)^2`), and at 4:2:0 four chroma TBs per component
    /// packed the same way at the quarter chroma size. Only for CUs of
    /// 16x16 and 32x32 in 4:0:0 and 4:2:0 (`InterPicture::tu_split`), so
    /// every child keeps `log2 > 2` and chroma splits with luma.
    /// `cbf_luma` and `cbf_chroma` are then the children's union — the
    /// depth-0 chroma bins the reader gates the children's on.
    pub split_tu: bool,
    /// Per quadrant under `split_tu`: `cbf_luma`.
    pub cbf_luma_tu: [bool; 4],
    /// Per component and quadrant under `split_tu`: `cbf_cb` / `cbf_cr`.
    pub cbf_chroma_tu: [[bool; 4]; 2],
}

impl InterCuDecision {
    /// Every field back to its default for a CU of `1 << log2_cu` decided
    /// under `ctx`, except the coefficient arrays, which keep whatever
    /// they held: each is written before it is read, over the blocks the
    /// cbfs then name.
    fn reset<S: Sample>(&mut self, log2_cu: u32, ctx: &MeCtx<'_, S>) {
        self.log2_cu = log2_cu;
        self.bypass = ctx.bypass;
        self.sign_hiding = ctx.sign_hiding;
        self.qp_y = ctx.qp;
        self.kind = InterCuKind::Skip { merge_idx: 0 };
        self.part = PartMode::P2Nx2N;
        self.pus = [PuPick::default(); 2];
        self.mv = Mv::ZERO;
        self.ref_idx = 0;
        self.mv_l1 = Mv::ZERO;
        self.ref_idx_l1 = -1;
        self.rqt_root_cbf = false;
        self.cbf_luma = false;
        self.cbf_chroma = [false; 2];
        self.cbf_chroma_bot = [false; 2];
        self.split_tu = false;
        self.cbf_luma_tu = [false; 4];
        self.cbf_chroma_tu = [[false; 4]; 2];
    }
}

impl Default for InterCuDecision {
    fn default() -> Self {
        InterCuDecision {
            log2_cu: 0,
            bypass: false,
            sign_hiding: false,
            qp_y: 26,
            kind: InterCuKind::Skip { merge_idx: 0 },
            part: PartMode::P2Nx2N,
            pus: [PuPick::default(); 2],
            mv: Mv::ZERO,
            ref_idx: 0,
            mv_l1: Mv::ZERO,
            ref_idx_l1: -1,
            rqt_root_cbf: false,
            cbf_luma: false,
            cbf_chroma: [false; 2],
            cbf_chroma_bot: [false; 2],
            luma: [0; 1024],
            chroma: [[0; 1024]; 2],
            split_tu: false,
            cbf_luma_tu: [false; 4],
            cbf_chroma_tu: [[false; 4]; 2],
        }
    }
}

/// How one CU of a P slice was coded — the seam the picture writer and
/// the deblocker consume, because a P slice holds CUs of both kinds.
///
/// [`InterPicture::code_ctu`] answers [`InterCuKind::UseIntra`] when the
/// flatness proxy says inter has lost; the caller then calls
/// [`InterPicture::code_ctu_intra`] and keeps *that* decision instead.
/// The inter one it displaces described nothing that was coded: its
/// coefficients were never quantised and its reconstruction never
/// written.
///
/// The intra half is boxed because it is the larger of the two by a
/// factor of two (a [`CuDecision`] carries the split shape's sixteen
/// coefficient slots) and it is the rarer by far; an unboxed enum would
/// double the memory of every P picture for the variant that seldom
/// fires.
// Inter is the common variant and stays inline; Intra is the rare one and is boxed.
#[allow(clippy::large_enum_variant)]
#[derive(Clone)]
pub enum PCuDecision {
    /// An inter CU: skip, merge or AMVP. Boxed, as the intra one is: the
    /// decision carries its coefficients, several kilobytes that the
    /// quadtree walk would otherwise copy at every return.
    Inter(Box<InterCuDecision>),
    /// An intra CU inside the P slice, decided by the intra module over
    /// this same picture's reconstruction.
    Intra(Box<CuDecision>),
}

impl PCuDecision {
    /// The `QpY` a decoder holds for this CU — see `InterCuDecision::qp_y`.
    pub fn qp_y(&self) -> i32 {
        match self {
            PCuDecision::Inter(d) => d.qp_y,
            PCuDecision::Intra(d) => d.qp_y,
        }
    }

    /// Store the `QpY` the encoder's quantiser chain settled on.
    pub fn set_qp_y(&mut self, qp_y: i32) {
        match self {
            PCuDecision::Inter(d) => d.qp_y = qp_y,
            PCuDecision::Intra(d) => d.qp_y = qp_y,
        }
    }

    /// Whether the CU carries any coded cbf, which is when — and only
    /// when — a decoder reads a `cu_qp_delta` inside it. An inter CU's
    /// tree exists exactly when `rqt_root_cbf` is set, and a tree at
    /// this geometry always carries a cbf (the decision spells a
    /// residual-free tree as a skip or a root cbf of 0).
    pub fn any_cbf(&self) -> bool {
        match self {
            PCuDecision::Inter(d) => d.rqt_root_cbf,
            PCuDecision::Intra(d) => d.any_cbf(),
        }
    }
}

/// Per-picture state of the P-picture walk: the reconstruction the next
/// CUs and the next picture predict from, and the decoder-grade side state
/// the candidate derivation reads. The caller walks CTUs in raster order
/// and calls `InterPicture::code_ctu` for each.
pub struct InterPicture<S: Sample> {
    /// Decoder-grade per-picture arrays: `merge_candidate` / `amvp` read
    /// z-scan availability (6.4.1) out of these, so this module maintains
    /// what the decoder maintains — `ctb_slice_addr` / `ctb_slice` marked
    /// as the walk reaches each CTB, `pred_mode` filled per CU.
    pub info: PicInfo,
    /// The reconstruction, decoder-identical by construction, with the
    /// per-4x4 motion grid filled exactly as `prediction_unit` fills it
    /// ("Store motion", src/hevc/ctu.rs) — `merge_candidate` / `amvp`
    /// read neighbour motion from here, and so does a later picture's
    /// temporal candidate when this one is its collocated picture.
    pub recon: Frame<S>,
    /// log2 of the CTB size, 4 or 5. A coding unit is this size or, under
    /// the coding quadtree, smaller — down to the 8x8 minimum coding block;
    /// the CU-level calls take their own size.
    pub log2_ctb: u32,
    /// `ChromaArrayType` (0 monochrome, 1 4:2:0, 2 4:2:2, 3 4:4:4), read
    /// off the parsed SPS. Every chroma decision below — whether chroma
    /// exists, where its transform blocks sit, which QP mapping applies —
    /// is gated on this exactly as the reader gates on
    /// `Sps::chroma_array_type`. Chroma *motion compensation* is not
    /// gated here at all: `predict_block` takes the subsampling from the
    /// frame it writes into, so building `recon` in the SPS's own format
    /// is what makes the chroma vector right (see `code_ctu`).
    pub cat: u32,
    /// The current picture's POC (the motion grid stores POC differences).
    pub cur_poc: i32,
    /// Whether a coded 16x16 or 32x32 CU may split its transform once
    /// (`InterCuDecision::split_tu`), decided in SSD plus lambda times
    /// the tree's bits against the unsplit TU.
    pub tu_split: bool,
    /// The largest CU, log2, the split is tried in.
    pub tu_split_max: u32,
    /// Scratch for the split trial: the prediction, the unsplit
    /// reconstruction (luma then both chroma planes, packed) and the split
    /// shape's levels.
    tu_save: Vec<S>,
    tu_keep: Vec<S>,
    tu_alt: Box<InterCuDecision>,
    /// Inter decisions the quadtree walk discarded, kept for the next
    /// unit rather than allocated, zeroed and copied each time.
    pool: Vec<Box<InterCuDecision>>,
    /// Trial saves finished with, for [`Self::take_save`].
    saves: Vec<TrialSave<S>>,
    /// The transform tree's price, as the last [`Self::code_residual_cu`]
    /// left the unit, when its split trial priced it.
    tree_bits: Option<f64>,
    /// The slice is B (the initial contexts a rate falls back to).
    pub slice_b: bool,
    /// Whether the chroma of an inter unit is quantised by RDOQ where the
    /// context asks for it (`MeCtx::rdoq`); otherwise only its luma is,
    /// the chroma by the plain quantiser. Intra units are not affected.
    pub rdoq_chroma: bool,
    /// `slice_temporal_mvp_enabled_flag`: the SPS enables temporal MVP and
    /// every inter slice this encoder writes then sets it, naming list 1's
    /// first picture for a B slice and list 0's for a P one.
    pub tmvp: bool,
    /// How deep [`InterPicture::code_ctu_intra`] may let an intra CU's
    /// transform tree split — the same knob as
    /// [`super::h265_intra::IntraPicture::split_depth`] and the same
    /// caveat: the coding-tree writer must be able to spell whichever
    /// shapes it permits, and a decision the writer cannot serialise
    /// desyncs the arithmetic coder. **1**, matching what the picture
    /// writer sets for I slices and what `write_cu_intra_body` spells.
    pub split_depth: u32,
    /// The prediction-unit shapes an inter CU may take besides 2Nx2N:
    /// `Config::inter_parts`, set by the caller. [`InterParts::None`]
    /// until then, which is every CU a single unit.
    pub parts: InterParts,
    /// The last CU decision's two candidates, as [`Self::code_cu`] or
    /// [`Self::code_cu_b`] left them: the 2Nx2N unit and the cheapest
    /// partitioned shape, if one was offered. `tree_leaf` codes the loser
    /// too when the two are close, and keeps the cheaper in SSD plus
    /// lambda times the bits of the whole CU.
    alts: Option<(PuPick, Option<PartsPick>)>,
    /// The half-sample planes of the references, by POC, when the caller
    /// has them (`h265_search::HalfPlanes`).
    pub(crate) half: Vec<(i32, std::sync::Arc<h265_search::HalfPlanes<S>>)>,
    /// How many times the full-sample search widens to its square rings
    /// before it stops (`search_full`).
    pub search_rings: u32,
    /// The best vector each (list, reference) found at each coding-tree
    /// depth, by the unit's position, for the units below it to start
    /// from: a child's motion is most often its parent's.
    seed_at: [[Option<(u32, u32, Mv)>; 16]; 4],
    /// Scratch for a bi-prediction scored from the fractional planes.
    spred_avg: Vec<S>,
    /// How many rounds of iterative bi-prediction refinement a B unit's
    /// AMVP pair gets ([`Self::refine_bi`]); 0 scores the pair of
    /// independently searched vectors only.
    pub bi_refine: u32,
    /// Refinement is tried only where the unrefined pair costs less than
    /// this multiple of the better one-list shape.
    pub bi_refine_gate: f32,
    /// The transform split is tried only where the whole TU kept at least
    /// this many luma levels.
    pub tu_split_nz: usize,
    /// How many of the merge candidates a unit scores, in list order.
    pub merge_tried: usize,
    /// The early split: a unit above the smallest whose 2Nx2N search
    /// costs more than this many SATD Lagrangians per sample, and is not
    /// a merge, is not coded whole — the walk goes straight to its four
    /// children (`tree_node`).
    pub split_gate: Option<f32>,
    /// The gate applies to units above this log2 size only.
    pub split_gate_above: u32,
    /// Whether the unit being decided may be gated ([`Self::split_gate`]).
    gate_now: bool,
    /// Whether it was.
    gated: bool,
    /// Scratch for [`Self::refine_bi`]: the other list's prediction, then
    /// the target the refined list is searched against.
    bi_target: Vec<S>,
    /// Whether a unit is offered intra by coding it (`Some`), or by the
    /// flat-block proxy [`prefer_intra`] (`None`).
    ///
    /// Under `Some(gate)` every unit is decided inter first and then, unless
    /// it is a skip or its cost per sample is below `gate` times lambda,
    /// coded intra as well, the cheaper kept. Measured 2026-11-15 against
    /// the proxy (x265 parity, 30 frames, `--preset medium`): the proxy
    /// chose intra almost never, and coding intra where the inter unit kept
    /// residual won 30-37% of those units on foreman and ducks — BD-rate
    /// against x265 medium went from -0.49% to -3.46% (ducks +16.2 to
    /// +10.9, foreman -6.7 to -12.4). Without the gate every unit tries:
    /// -2.8% before the search work, at 1.7x the CPU; skips win intra 448
    /// times in 338,000 trials and units under half a lambda per sample 896
    /// in 235,000, which is what the gate leaves out (gate 1.0: -2.83%).
    pub rd_intra: Option<f32>,
    /// Recursion skip: a unit coded as a skip whose cost is below this
    /// many lambdas per sample is not tried split. `None` tries every
    /// split. Measured 2026-11-15 (x265 parity, `--preset medium`, CPU by
    /// interleaved one-thread runs on park_joy 720p and foreman CIF at QP
    /// 32): every skip, -2.55% BD-rate where -4.37% was, at 0.82x the CPU;
    /// below 1.0 lambda -3.00%, 0.5 -3.77%, 0.25 -4.28% at 0.89x.
    pub rskip: Option<f32>,
    /// Intra is coded only where the best intra mode's SATD cost comes
    /// within this factor of the inter unit's search cost; `None` codes it
    /// wherever `rd_intra` lets it.
    pub intra_satd_gate: Option<f32>,
    /// The SATD-domain search cost of the last inter unit decided.
    last_pick_cost: f32,
    /// Search only the nearest reference and the parent unit's below the
    /// top of the coding tree (`ref_searched`).
    pub limit_refs: bool,
    /// The reference each depth's whole unit chose, by position.
    parent_ref: [Option<(u32, u32, i8)>; 4],
    /// Whether a unit with residual is also tried without it
    /// ([`Self::tree_leaf_drop`]).
    pub drop_trial: bool,
    /// At the minimum coding block an intra unit also tries `PART_NxN`
    /// when its 2Nx2N cost is below this many times the inter unit's; 0
    /// never tries.
    pub nxn_trial: f64,
    /// The quarter-sample refinement scores the four axial neighbours only.
    pub quarter_cross: bool,
    /// Set for one call of `code_whole`: predict, but code no residual —
    /// the rate-distortion trial of a unit without its coefficients.
    no_residual: bool,
    /// The picture descriptor the intra decision's availability and MPM
    /// mirrors read, built once from the SPS.
    geo: Geo,
    /// Reference-sample scratch for the intra decision, held per picture
    /// exactly as [`super::h265_intra::IntraPicture`] holds its own.
    intra_scratch: IntraScratch,
    /// MC scratch, as the decoder allocates per slice.
    scratch: McScratch<S>,
    /// Luma-only prediction scratch for candidate scoring: the clamp
    /// window, the two-stage filter intermediate, the 14-bit prediction
    /// and the sample-domain prediction.
    swin: Vec<S>,
    stmp: Vec<i16>,
    spred14: Vec<i16>,
    /// The second list's 14-bit prediction, for the B bi-prediction trial
    /// (`satd_bi_at`). Unused by the P path.
    spred14_b: Vec<i16>,
    spred: Vec<S>,
    /// The explicit weighting this picture's list-0 predictions carry,
    /// one per reference in `RefPicList0` — per component, what
    /// `hevc::ctu::explicit_weighting` derives from the slice's
    /// `pred_weight_table` for that reference. Empty when the slice
    /// carries no table (every reference default). Read by
    /// [`Self::code_ctu`]'s scoring and its final prediction, so a
    /// vector is chosen for the prediction that will actually be made.
    /// The B walk does not read it; see [`Self::wp_b`].
    pub wp: Vec<[Weighting; 3]>,
    /// The explicit weighting a B picture's predictions carry, by
    /// `inter_pred_idc` — `[0]` `PRED_L0`, `[1]` `PRED_L1`, `[2]` `PRED_BI`
    /// — per component: what `hevc::ctu::explicit_weighting` derives from
    /// the slice's `pred_weight_table` for reference indices `[0, -1]`,
    /// `[-1, 0]` and `[0, 0]`, each list's one active reference. All
    /// `Weighting::Default` when the slice carries no table, or one whose
    /// every entry is the default (which predicts the same samples). Read
    /// by [`Self::code_cu_b`]'s search, its merge and bi scoring and its
    /// final prediction. The P walk does not read it.
    pub wp_b: [[Weighting; 3]; 3],
    /// The luma plane each list's full-sample search scores against in a
    /// weighted B picture: the list's reference with its luma weighting
    /// applied sample by sample, or `None` where the list's luma is
    /// unweighted and the search reads the reference itself. The search is
    /// a SAD over whole samples, and a reference a gain or an offset away
    /// from the picture flattens that landscape — an offset of 24 on the
    /// test grating defeats it outright — so a weighted list searches what
    /// it will predict from. Set by [`Self::set_b_weights`]; the
    /// sub-sample refinement and the prediction still read the reference.
    pub search_b: [Option<Plane16<S>>; 2],
}

impl<S: Sample> InterPicture<S> {
    /// State for one P picture, from the *parsed* parameter sets — the
    /// caller round-trips the bytes `write_sps` / `write_pps` produced
    /// through the decoder's own parsers, the same proof the encoder
    /// applies to everything it writes. Fixed geometry as in the intra
    /// module: whole CTUs, 4:2:0, `log2_cu` 4 or 5.
    pub fn new(sps: &Sps, pps: &Pps, cur_poc: i32) -> Self {
        assert!(
            (4..=5).contains(&sps.log2_ctb_size),
            "log2_ctb {} outside 4..=5",
            sps.log2_ctb_size
        );
        let (w, h) = (sps.width as usize, sps.height as usize);
        // Whole minimum coding blocks; edge CTBs may be partial (see
        // `tree_node`).
        let m = 1usize << MIN_CB_LOG2;
        assert!(
            w.is_multiple_of(m) && h.is_multiple_of(m),
            "{w}x{h} is not a whole number of {m}x{m} coding blocks"
        );
        let geo = std::sync::Arc::new(Geometry::new(sps, pps));
        let info = PicInfo::new(geo);
        // The reconstruction is built in the SPS's own chroma format, and
        // that single choice is what gives chroma motion compensation its
        // per-format behaviour: `predict_block` reads `cur.chroma`, derives
        // (SubWidthC, SubHeightC) from it and scales the vector by
        // `mv * 2 / SubWidthC` per axis (its `mvc` closure, 8.5.3.2.10) —
        // so 4:2:2's unscaled-horizontal, doubled-vertical chroma vector is
        // the decoder's derivation, not one retyped here.
        let mut recon = Frame::new(w, h, sps.chroma_format(), sps.bit_depth_luma);
        recon.poc = cur_poc;
        let nmax = 1usize << (2 * sps.log2_ctb_size);
        InterPicture {
            info,
            recon,
            log2_ctb: sps.log2_ctb_size,
            cat: sps.chroma_array_type(),
            cur_poc,
            split_depth: 1,
            parts: InterParts::None,
            alts: None,
            no_residual: false,
            rd_intra: None,
            rskip: None,
            drop_trial: true,
            limit_refs: false,
            parent_ref: [None; 4],
            intra_satd_gate: None,
            last_pick_cost: 0.0,
            nxn_trial: 2.0,
            quarter_cross: false,
            half: Vec::new(),
            search_rings: 0,
            seed_at: [[None; 16]; 4],
            spred_avg: vec![S::default(); 64 * 64],
            bi_refine: 0,
            bi_refine_gate: f32::INFINITY,
            tu_split_nz: 0,
            merge_tried: MAX_MERGE_CAND,
            split_gate: None,
            split_gate_above: 3,
            gate_now: false,
            gated: false,
            bi_target: vec![S::default(); 64 * 64],
            geo: Geo::new(sps.log2_ctb_size, w, h, sps.chroma_array_type()),
            intra_scratch: IntraScratch::default(),
            scratch: McScratch::new(),
            swin: vec![S::default(); (64 + 7) * (64 + 7)],
            stmp: vec![0; crate::dsp::hevc::MC_TMP_LEN],
            spred14: vec![0; nmax],
            spred14_b: vec![0; nmax],
            spred: vec![S::default(); nmax],
            wp: Vec::new(),
            wp_b: [[Weighting::Default; 3]; 3],
            search_b: [None, None],
            tmvp: sps.temporal_mvp_enabled,
            tu_split: false,
            tu_split_max: 5,
            tu_save: Vec::new(),
            tu_keep: Vec::new(),
            tu_alt: Box::default(),
            pool: Vec::new(),
            saves: Vec::new(),
            tree_bits: None,
            rdoq_chroma: true,
            slice_b: false,
        }
    }

    /// The weighting list-0 reference `r` carries: the table's, or the
    /// default when the slice has no table.
    fn wp_for(&self, r: usize) -> [Weighting; 3] {
        self.wp.get(r).copied().unwrap_or([Weighting::Default; 3])
    }

    /// Weight this B picture's predictions by its slice's table `t`, whose
    /// list-0 and list-1 references are `ref0` and `ref1`: each
    /// `inter_pred_idc`'s entry of [`Self::wp_b`] as the reader derives it
    /// — `explicit_weighting` for reference 0 of the lists that
    /// prediction uses, `[0, -1]`, `[-1, 0]` and `[0, 0]` — and, for each
    /// list whose luma entry is not the default, the plane its full-sample
    /// search scores against ([`Self::search_b`]).
    pub fn set_b_weights(
        &mut self,
        t: &PredWeightTable,
        bit_depth_luma: u32,
        bit_depth_chroma: u32,
        ref0: &Frame<S>,
        ref1: &Frame<S>,
    ) {
        self.wp_b = [[0, -1], [-1, 0], [0, 0]]
            .map(|r| explicit_weighting(t, bit_depth_luma, bit_depth_chroma, r));
        for (list, rf) in [ref0, ref1].into_iter().enumerate() {
            let (w, o) = t.lists[list][0].luma;
            self.search_b[list] = match self.wp_b[list][0] {
                Weighting::Explicit { log2_wd, .. } if (w, o) != (1 << t.luma_log2_denom, 0) => {
                    Some(weighted_search_plane(&rf.y, bit_depth_luma, log2_wd, w, o))
                }
                _ => None,
            };
        }
    }

    /// The weighting a B prediction from the lists `ref_idx` uses carries
    /// (see [`Self::wp_b`]): one list's, or the pair's.
    fn wp_b_for(&self, ref_idx: [i8; 2]) -> [Weighting; 3] {
        match (ref_idx[0] >= 0, ref_idx[1] >= 0) {
            (true, false) => self.wp_b[0],
            (false, true) => self.wp_b[1],
            _ => self.wp_b[2],
        }
    }

    /// The reference-list context the decoder's candidate derivation
    /// takes, for a P slice whose `RefPicList0` holds `pocs0`, nearest
    /// first, with `col` (the list's first picture) collocated when the
    /// stream enables temporal MVP (see the module header).
    fn ref_ctx<'a>(&self, pocs0: &[i32], col: &'a Frame<S>) -> RefCtx<'a, S> {
        RefCtx {
            pocs: [pocs0.to_vec(), Vec::new()],
            long_term: [vec![false; pocs0.len()], Vec::new()],
            col: self.tmvp.then_some(col),
            cur_poc: self.cur_poc,
            no_backward_pred: true,
            tmvp: self.tmvp,
            max_merge_cand: MAX_MERGE_CAND,
            log2_par_mrg_level: 2,
            is_b: false,
            num_ref_idx: [pocs0.len(), 0],
            col_from_l0: true,
        }
    }

    /// The reference-list context for a B slice referencing `poc0` in
    /// list 0 and `poc1` in list 1, one active entry each — what the
    /// header this stream carries declares (`num_ref_idx_active_override`
    /// 0 over PPS defaults of 0, which the reader resolves as [1, 1]).
    ///
    /// `no_backward_pred` is **derived** exactly as `decoder.rs` derives
    /// `NoBackwardPredFlag` — every reference POC at or before the current
    /// picture — rather than pasted from [`Self::ref_ctx`], which
    /// hardcodes true because a P slice's single reference is always in
    /// the past. For a B between two anchors it comes out false, and it
    /// reaches `merge_candidate` through `RefCtx`.
    fn ref_ctx_b<'a>(&self, poc0: i32, poc1: i32, col: &'a Frame<S>) -> RefCtx<'a, S> {
        let no_backward_pred = [poc0, poc1].iter().all(|&p| p <= self.cur_poc);
        RefCtx {
            pocs: [vec![poc0], vec![poc1]],
            long_term: [vec![false], vec![false]],
            col: self.tmvp.then_some(col),
            cur_poc: self.cur_poc,
            no_backward_pred,
            tmvp: self.tmvp,
            max_merge_cand: MAX_MERGE_CAND,
            log2_par_mrg_level: 2,
            is_b: true,
            num_ref_idx: [1, 1],
            // The collocated picture is list 1's (the future anchor).
            col_from_l0: false,
        }
    }

    /// Decide and code one CTU as a single 2Nx2N CU against the references
    /// of `RefPicList0`, `refs_l0`, nearest first (borders extended —
    /// `Frame::extend_rows` — exactly as the decoder pads references
    /// before MC reads them).
    ///
    /// On [`InterCuKind::UseIntra`] the reconstruction planes are
    /// untouched and the motion grid holds `MotionInfo::INTRA`: the
    /// caller runs the intra decision, which writes the planes itself.
    ///
    /// `src_cb` / `src_cr` are the chroma planes at `c_stride`, in this
    /// stream's own format — `(width / SubWidthC) x (height / SubHeightC)`.
    /// Under monochrome they are never read and may be empty, exactly as
    /// the reader never reads a chroma element when `chroma_array_type`
    /// is 0.
    #[allow(clippy::too_many_arguments)]
    pub fn code_ctu(
        &mut self,
        ctx: &MeCtx<'_, S>,
        refs_l0: &[&Frame<S>],
        cu_x: usize,
        cu_y: usize,
        src_y: &[S],
        y_stride: usize,
        src_cb: &[S],
        src_cr: &[S],
        c_stride: usize,
    ) -> InterCuDecision {
        let n = 1usize << self.log2_ctb;
        let mut out = InterCuDecision::default();
        self.code_cu_into(
            ctx,
            refs_l0,
            cu_x * n,
            cu_y * n,
            self.log2_ctb,
            0,
            src_y,
            y_stride,
            src_cb,
            src_cr,
            c_stride,
            &mut out,
        );
        out
    }

    /// [`Self::code_ctu`] for one coding unit of `1 << log2_cu` at luma
    /// `(x0, y0)` and coding-tree depth `depth` — the whole CTB at depth 0,
    /// a quadtree leaf below it. Everything the decision derives from its
    /// neighbours (merge and AMVP candidates, the skip context) comes from
    /// the decoder-grade state the walk maintains, so a neighbour of any
    /// size serves; `depth` reaches the rate model only through
    /// `inter_pred_idc`'s context.
    #[allow(clippy::too_many_arguments)]
    pub fn code_cu_into(
        &mut self,
        ctx: &MeCtx<'_, S>,
        refs_l0: &[&Frame<S>],
        x0: usize,
        y0: usize,
        log2_cu: u32,
        depth: u32,
        src_y: &[S],
        y_stride: usize,
        src_cb: &[S],
        src_cr: &[S],
        c_stride: usize,
        out: &mut InterCuDecision,
    ) {
        let n = 1usize << log2_cu;
        out.reset(log2_cu, ctx);

        // Mark the CTB as this (single) slice's, as the decoder does at CTB
        // start: `avail_ctx` reads the current CTB's slice address, and
        // `available_at` compares neighbours against it.
        let ctb = self.info.ctb_of(x0, y0);
        self.info.ctb_slice_addr[ctb] = 0;
        self.info.ctb_slice[ctb] = 0;

        debug_assert!(!refs_l0.is_empty(), "a P CU needs at least one reference");
        let pu = PuPos {
            x_cb: x0 as i32,
            y_cb: y0 as i32,
            n_cb: n as i32,
            x_pb: x0 as i32,
            y_pb: y0 as i32,
            w: n as i32,
            h: n as i32,
            part_idx: 0,
        };
        let lam = lambda(ctx.qp) * satd_lambda_scale(ctx.bit_depth);
        let rate = Rate::new_at(ctx.qp, false, log2_cu, depth);
        let pick = self.pick_pu(ctx, refs_l0, &pu, src_y, y_stride, &rate, lam);
        if self.gate_now
            && pick.merge_idx.is_none()
            && self
                .split_gate
                .is_some_and(|t| pick.cost > t * lam * (n * n) as f32)
        {
            self.gated = true;
            return;
        }
        let parts = self.pick_parts(
            ctx,
            TreeRefs::P(refs_l0),
            x0,
            y0,
            log2_cu,
            src_y,
            y_stride,
            &rate,
            lam,
        );
        self.alts = Some((pick, parts));
        self.last_pick_cost = parts.map_or(pick.cost, |p| p.2.min(pick.cost));
        if let Some((part, pus, _)) = parts.filter(|p| p.2 < pick.cost) {
            self.code_parts(
                ctx,
                TreeRefs::P(refs_l0),
                out,
                part,
                pus,
                x0,
                y0,
                log2_cu,
                src_y,
                y_stride,
                src_cb,
                src_cr,
                c_stride,
            );
            return;
        }
        self.code_whole(
            ctx,
            TreeRefs::P(refs_l0),
            out,
            pick,
            x0,
            y0,
            log2_cu,
            src_y,
            y_stride,
            src_cb,
            src_cr,
            c_stride,
        );
    }

    /// The motion decision for one prediction unit `pu` of a P CU: the
    /// cheaper of merge, over the decoder's own candidates, and AMVP, over
    /// every reference in `refs_l0`, by luma SATD plus `lam` times the bits
    /// `rate` prices each shape at. The 2Nx2N CU's single PU is `pu` with
    /// the CU's own rectangle. Reads the decoder-grade state the walk
    /// maintains and writes none of it.
    #[allow(clippy::too_many_arguments)]
    fn pick_pu(
        &mut self,
        ctx: &MeCtx<'_, S>,
        refs_l0: &[&Frame<S>],
        pu: &PuPos,
        src_y: &[S],
        y_stride: usize,
        rate: &Rate,
        lam: f32,
    ) -> PuPick {
        let (x0, y0) = (pu.x_pb as usize, pu.y_pb as usize);
        let wh = (pu.w as usize, pu.h as usize);
        let whole = pu.w == pu.n_cb && pu.h == pu.n_cb;
        let nref = refs_l0.len() as u32;
        let pocs: Vec<i32> = refs_l0.iter().map(|f| f.poc).collect();
        let refs = self.ref_ctx(&pocs, refs_l0[0]);
        // The decoder's own candidate lists (see the module header). The
        // AMVP predictors depend on the reference index — the derivation
        // scales a neighbour's vector by the POC-distance ratio between
        // its reference and this one — so they are derived per
        // reference by calling `amvp` once for each.
        let merge: [Cand; MAX_MERGE_CAND] = merge_candidates(&self.info, &self.recon, &refs, pu);
        let mvp: Vec<[Mv; 2]> = (0..refs_l0.len())
            .map(|r| {
                [
                    amvp(&self.info, &self.recon, &refs, pu, 0, r as i8, 0),
                    amvp(&self.info, &self.recon, &refs, pu, 0, r as i8, 1),
                ]
            })
            .collect();

        let src = &src_y[y0 * y_stride + x0..];

        // One search per reference. Multi-reference is not a wider search
        // of the same picture but a CHOICE between pictures: a block
        // uncovered by motion, or one whose match is periodic, can be
        // predicted better from an older frame than the newest one, and
        // the only way to know which is to search each and price the
        // answer — `ref_idx` is signalled per prediction unit, so the
        // choice is not free and the cost below counts it.
        //
        // The default is ONE reference, and that is a measured choice.
        // On whole-CTB units (`max_cu_depth` 0) the choice never has a
        // better answer: 32x32 coding units average over enough
        // content that one reference always serves them (0 of 140 blocks
        // across the corpus, where the same probe at 16x16 said 6.9% and
        // was answering about a block size this encoder does not code),
        // and two references measured 0.80% worse on every clip, better
        // on none. `Config::max_refs` opts in. The coding quadtree now codes
        // the 16x16 and 8x8 units that probe answered about; it has not
        // been re-run against them.
        //
        // Full-sample descent from every distinct seed's best, then the
        // two sub-sample rings, per reference.
        let mut per_ref: Vec<(Mv, u32)> = Vec::with_capacity(refs_l0.len());
        for (r, rf) in refs_l0.iter().enumerate() {
            if !self.ref_searched(pu, r) {
                per_ref.push((Mv::ZERO, u32::MAX));
                continue;
            }
            let mut seeds = [Mv::ZERO; 3 + MAX_MERGE_CAND + 1];
            let mut ns = 3;
            (seeds[1], seeds[2]) = (mvp[r][0], mvp[r][1]);
            for c in merge.iter().filter(|c| c.ref_idx[0] == r as i8) {
                seeds[ns] = c.mv[0];
                ns += 1;
            }
            if let Some(p) = self.parent_seed(pu, 0, r) {
                seeds[ns] = p;
                ns += 1;
            }
            let full = self.search_full(
                ctx,
                &rf.y,
                x0,
                y0,
                wh,
                src,
                y_stride,
                &seeds[..ns],
                lam,
                mvp[r],
            );
            let wp = self.wp_for(r)[0];
            let half = self.half_for(rf.poc);
            let found = self.refine_subpel(
                ctx,
                &rf.y,
                half.as_deref(),
                wp,
                0,
                x0,
                y0,
                wh,
                src,
                y_stride,
                full,
                lam,
                mvp[r],
            );
            self.note_seed(pu, 0, r, found.0);
            per_ref.push(found);
        }

        // Cost the shapes. Merge candidates are scored at their exact
        // vectors against the picture they actually name; only the first
        // occurrence of a (vector, reference) matters (a later duplicate
        // signals strictly more bins for the same prediction).
        let mut best_merge: Option<(usize, u32)> = None; // (idx, satd)
        let mut best_merge_cost = f32::INFINITY;
        let mut seen: Vec<(Mv, i8)> = Vec::with_capacity(MAX_MERGE_CAND);
        for (idx, cand) in merge.iter().enumerate().take(self.merge_tried) {
            let ri = cand.ref_idx[0];
            if ri < 0 || ri as usize >= refs_l0.len() || seen.contains(&(cand.mv[0], ri)) {
                continue;
            }
            seen.push((cand.mv[0], ri));
            let satd = self.satd_at(
                ctx,
                &refs_l0[ri as usize].y,
                ri as usize,
                x0,
                y0,
                wh,
                src,
                y_stride,
                cand.mv[0],
            );
            // cu_skip_flag or merge_flag, plus the TR-coded index.
            // Skip and merge differ in signalling, and a zero-residual
            // candidate becomes a skip, so price each at the shape it
            // would actually take. One unit of two is never a skip: it
            // spells merge_flag and the index, and the CU around it is
            // priced once, by the caller.
            let bits = if whole {
                rate.skip(idx as u8).min(rate.merge(idx as u8))
            } else {
                rate.pu(
                    &PuPick {
                        merge_idx: Some(idx as u8),
                        ..PuPick::default()
                    },
                    false,
                    pu,
                    nref,
                )
            };
            let cost = satd as f32 + lam * bits;
            if cost < best_merge_cost {
                best_merge_cost = cost;
                best_merge = Some((idx, satd));
            }
        }
        // AMVP, over every reference: its searched vector against the
        // cheaper of its own two predictors (the reader's uLX sum is a
        // wrapping add, so the mvd is a wrapping difference), plus what
        // it costs to say WHICH picture — `ref_idx` is truncated unary
        // over the active count and absent entirely at one reference,
        // so a single-reference stream prices exactly as it always did.
        let mut best_amvp: Option<(usize, u8, Mv, u32)> = None; // (ref, flag, mvd, satd)
        let mut amvp_cost = f32::INFINITY;
        for (r, &(mv_r, satd_r)) in per_ref.iter().enumerate() {
            let mvd_for = |p: Mv| Mv::new(mv_r.x.wrapping_sub(p.x), mv_r.y.wrapping_sub(p.y));
            let (d0, d1) = (mvd_for(mvp[r][0]), mvd_for(mvp[r][1]));
            let price = |d: Mv, flag: u8| {
                if whole {
                    rate.amvp_ref(d, flag, true, nref, r as u32)
                } else {
                    rate.pu(
                        &PuPick {
                            mvd: [d, Mv::ZERO],
                            mvp_flag: [flag, 0],
                            ref_idx: [r as i8, -1],
                            ..PuPick::default()
                        },
                        false,
                        pu,
                        nref,
                    )
                }
            };
            let c0 = price(d0, 0);
            let c1 = price(d1, 1);
            let (flag, mvd, bits) = if c1 < c0 {
                (1u8, d1, c1)
            } else {
                (0u8, d0, c0)
            };
            // merge_flag 0, ref_idx, mvp_l0_flag, rqt_root_cbf, plus the
            // mvd bins (cu_skip_flag and pred_mode/part_mode surround
            // both shapes).
            let cost = satd_r as f32 + lam * bits;
            if cost < amvp_cost {
                amvp_cost = cost;
                best_amvp = Some((r, flag, mvd, satd_r));
            }
        }
        let (amvp_ref, amvp_flag, mvd, amvp_satd) = best_amvp.expect("at least one reference");

        match best_merge {
            Some((idx, satd)) if best_merge_cost <= amvp_cost => PuPick {
                merge_idx: Some(idx as u8),
                idc: 0,
                mvd: [Mv::ZERO; 2],
                mvp_flag: [0; 2],
                mv: [merge[idx].mv[0], Mv::ZERO],
                ref_idx: [merge[idx].ref_idx[0].max(0), -1],
                satd,
                cost: best_merge_cost,
            },
            _ => PuPick {
                merge_idx: None,
                idc: 0,
                mvd: [mvd, Mv::ZERO],
                mvp_flag: [amvp_flag, 0],
                mv: [per_ref[amvp_ref].0, Mv::ZERO],
                ref_idx: [amvp_ref as i8, -1],
                satd: amvp_satd,
                cost: amvp_cost,
            },
        }
    }

    /// Decide and code one CTU as a single 2Nx2N CU of a **B** picture
    /// against `ref0` (list 0) and `ref1` (list 1), whose borders must be
    /// extended as the decoder pads references before MC reads them.
    ///
    /// The shape of the decision, and what is the decoder's rather than
    /// this module's:
    ///
    /// - **The candidate lists are the decoder's.** `merge_candidate` runs
    ///   with `is_b` true, so it returns both lists per candidate and adds
    ///   the combined bi-predictive pairs and the bi zero candidates that
    ///   only a B slice has. `amvp` runs per list. Neither derivation is
    ///   mirrored here.
    /// - **The search is per list**, each the same greedy full-sample
    ///   descent plus two sub-sample rings the P walk uses, seeded from
    ///   that list's own predictors and every merge candidate that uses
    ///   that list.
    /// - **One bi trial, at the two per-list winners**, scored through the
    ///   decoder's `dsp.bi` (`satd_bi_at`). There is deliberately **no
    ///   iterative bi refinement** in this version: the two vectors are
    ///   not re-searched against each other's prediction, so a bi CU here
    ///   is the best pair of independently searched vectors and not the
    ///   best pair. Named because it is a quality ceiling, not a
    ///   correctness one.
    /// - **The prediction is the decoder's own**, uni or bi, through the
    ///   one `predict_block` call that also gives every chroma format its
    ///   vector — so B costs nothing extra outside 4:2:0.
    ///
    /// On [`InterCuKind::UseIntra`] the reconstruction planes are
    /// untouched, exactly as in [`Self::code_ctu`].
    #[allow(clippy::too_many_arguments)]
    pub fn code_ctu_b(
        &mut self,
        ctx: &MeCtx<'_, S>,
        ref0: &Frame<S>,
        ref1: &Frame<S>,
        cu_x: usize,
        cu_y: usize,
        src_y: &[S],
        y_stride: usize,
        src_cb: &[S],
        src_cr: &[S],
        c_stride: usize,
    ) -> InterCuDecision {
        let n = 1usize << self.log2_ctb;
        let mut out = InterCuDecision::default();
        self.code_cu_b_into(
            ctx,
            ref0,
            ref1,
            cu_x * n,
            cu_y * n,
            self.log2_ctb,
            0,
            src_y,
            y_stride,
            src_cb,
            src_cr,
            c_stride,
            &mut out,
        );
        out
    }

    /// [`Self::code_ctu_b`] for one coding unit of `1 << log2_cu` at luma
    /// `(x0, y0)` and coding-tree depth `depth`, as [`Self::code_cu`] is
    /// to [`Self::code_ctu`].
    #[allow(clippy::too_many_arguments)]
    pub fn code_cu_b_into(
        &mut self,
        ctx: &MeCtx<'_, S>,
        ref0: &Frame<S>,
        ref1: &Frame<S>,
        x0: usize,
        y0: usize,
        log2_cu: u32,
        depth: u32,
        src_y: &[S],
        y_stride: usize,
        src_cb: &[S],
        src_cr: &[S],
        c_stride: usize,
        out: &mut InterCuDecision,
    ) {
        let n = 1usize << log2_cu;
        out.reset(log2_cu, ctx);

        let ctb = self.info.ctb_of(x0, y0);
        self.info.ctb_slice_addr[ctb] = 0;
        self.info.ctb_slice[ctb] = 0;

        let pu = PuPos {
            x_cb: x0 as i32,
            y_cb: y0 as i32,
            n_cb: n as i32,
            x_pb: x0 as i32,
            y_pb: y0 as i32,
            w: n as i32,
            h: n as i32,
            part_idx: 0,
        };
        let lam = lambda(ctx.qp) * satd_lambda_scale(ctx.bit_depth);
        let rate = Rate::new_at(ctx.qp, true, log2_cu, depth);
        let pick = self.pick_pu_b(ctx, ref0, ref1, &pu, src_y, y_stride, &rate, lam);
        if self.gate_now
            && pick.merge_idx.is_none()
            && self
                .split_gate
                .is_some_and(|t| pick.cost > t * lam * (n * n) as f32)
        {
            self.gated = true;
            return;
        }
        let parts = self.pick_parts(
            ctx,
            TreeRefs::B(ref0, ref1),
            x0,
            y0,
            log2_cu,
            src_y,
            y_stride,
            &rate,
            lam,
        );
        self.alts = Some((pick, parts));
        self.last_pick_cost = parts.map_or(pick.cost, |p| p.2.min(pick.cost));
        if let Some((part, pus, _)) = parts.filter(|p| p.2 < pick.cost) {
            self.code_parts(
                ctx,
                TreeRefs::B(ref0, ref1),
                out,
                part,
                pus,
                x0,
                y0,
                log2_cu,
                src_y,
                y_stride,
                src_cb,
                src_cr,
                c_stride,
            );
            return;
        }
        self.code_whole(
            ctx,
            TreeRefs::B(ref0, ref1),
            out,
            pick,
            x0,
            y0,
            log2_cu,
            src_y,
            y_stride,
            src_cb,
            src_cr,
            c_stride,
        );
    }

    /// [`Self::pick_pu`] for one prediction unit of a B CU: merge over the
    /// decoder's own candidates (both lists, the combined bi pairs and the
    /// bi zero candidates a B slice adds) against AMVP in list 0, list 1 or
    /// both, each list searched on its own and the bi trial scored at the
    /// two winners. Writes no decoder-grade state either.
    #[allow(clippy::too_many_arguments)]
    fn pick_pu_b(
        &mut self,
        ctx: &MeCtx<'_, S>,
        ref0: &Frame<S>,
        ref1: &Frame<S>,
        pu: &PuPos,
        src_y: &[S],
        y_stride: usize,
        rate: &Rate,
        lam: f32,
    ) -> PuPick {
        let (x0, y0) = (pu.x_pb as usize, pu.y_pb as usize);
        let wh = (pu.w as usize, pu.h as usize);
        let whole = pu.w == pu.n_cb && pu.h == pu.n_cb;
        let refs = self.ref_ctx_b(ref0.poc, ref1.poc, ref1);
        let merge: [Cand; MAX_MERGE_CAND] = merge_candidates(&self.info, &self.recon, &refs, pu);
        let mvp: [[Mv; 2]; 2] = [
            [
                amvp(&self.info, &self.recon, &refs, pu, 0, 0, 0),
                amvp(&self.info, &self.recon, &refs, pu, 0, 0, 1),
            ],
            [
                amvp(&self.info, &self.recon, &refs, pu, 1, 0, 0),
                amvp(&self.info, &self.recon, &refs, pu, 1, 0, 1),
            ],
        ];

        let src = &src_y[y0 * y_stride + x0..];

        // Per-list search. Each list is seeded from the zero vector, its
        // own two AMVP predictors, and every merge candidate that uses it.
        let mut uni = [(Mv::ZERO, u32::MAX); 2];
        for list in 0..2usize {
            let plane = if list == 0 { &ref0.y } else { &ref1.y };
            let mut seeds = [Mv::ZERO; 3 + MAX_MERGE_CAND + 1];
            let mut ns = 3;
            (seeds[1], seeds[2]) = (mvp[list][0], mvp[list][1]);
            for c in merge.iter().filter(|c| c.ref_idx[list] == 0) {
                seeds[ns] = c.mv[list];
                ns += 1;
            }
            // Whole samples are searched on the weighted reference where the
            // list is weighted (`search_b`), the sub-sample rings below on
            // the reference itself under the weighted scoring.
            if let Some(p) = self.parent_seed(pu, list, 0) {
                seeds[ns] = p;
                ns += 1;
            }
            let seeds = &seeds[..ns];
            let full = self.search_full(
                ctx,
                self.search_b[list].as_ref().unwrap_or(plane),
                x0,
                y0,
                wh,
                src,
                y_stride,
                seeds,
                lam,
                mvp[list],
            );
            // Scored under the weighting a one-list prediction from this
            // list carries, so the vector is chosen for the prediction
            // that will be made.
            let wp = self.wp_b[list][0];
            let half = self.half_for(if list == 0 { ref0.poc } else { ref1.poc });
            uni[list] = self.refine_subpel(
                ctx,
                plane,
                half.as_deref(),
                wp,
                list,
                x0,
                y0,
                wh,
                src,
                y_stride,
                full,
                lam,
                mvp[list],
            );
            self.note_seed(pu, list, 0, uni[list].0);
        }

        // The three AMVP shapes. `inter_pred_idc` costs two bins for a uni
        // shape and one for BI (the reader stops after a set first bin),
        // on top of merge_flag, rqt_root_cbf and a mvp_flag per used list
        // — the same approximate-bin-count placeholder policy as the rest
        // of this module.
        let mvd_for = |mv: Mv, p: Mv| Mv::new(mv.x.wrapping_sub(p.x), mv.y.wrapping_sub(p.y));
        // A whole CU's AMVP syntax, or one unit's of two.
        let amvp_bits = |idc: u8, mvd: [Mv; 2], fl: [u8; 2]| {
            if whole {
                rate.amvp_b(idc, mvd, fl, true)
            } else {
                rate.pu(
                    &PuPick {
                        idc,
                        mvd,
                        mvp_flag: fl,
                        ..PuPick::default()
                    },
                    true,
                    pu,
                    1,
                )
            }
        };
        let mut best_mvd = [Mv::ZERO; 2];
        let mut best_flag = [0u8; 2];
        for list in 0..2usize {
            let a = mvd_for(uni[list].0, mvp[list][0]);
            let b = mvd_for(uni[list].0, mvp[list][1]);
            let idc_for = |l: usize| if l == 0 { 0u8 } else { 1 };
            let one = |m: Mv, f: u8| {
                let mut mvd = [Mv::ZERO; 2];
                mvd[list] = m;
                let mut fl = [0u8; 2];
                fl[list] = f;
                amvp_bits(idc_for(list), mvd, fl)
            };
            let pick1 = one(b, 1) < one(a, 0);
            best_flag[list] = u8::from(pick1);
            best_mvd[list] = if pick1 { b } else { a };
        }
        // idc 0 / 1: one list, its mvd and mvp_flag.
        let mut best: Option<(u8, u32)> = None; // (idc, satd)
        let mut best_cost = f32::INFINITY;
        for (list, u) in uni.iter().enumerate() {
            let mut mvd = [Mv::ZERO; 2];
            mvd[list] = best_mvd[list];
            let mut fl = [0u8; 2];
            fl[list] = best_flag[list];
            let bits = amvp_bits(list as u8, mvd, fl);
            let cost = u.1 as f32 + lam * bits;
            if cost < best_cost {
                best_cost = cost;
                best = Some((list as u8, u.1));
            }
        }
        // idc 2: the bi trial at the two winners — not for an 8x4 or 4x8
        // unit, which the reader gives no PRED_BI spelling (its
        // `inter_pred_idc` codes one bin, L0 or L1, where `w + h == 12`).
        let bi_ok = pu.w + pu.h != 12;
        let mut bi_satd = if bi_ok {
            self.satd_bi_at(
                ctx,
                &ref0.y,
                &ref1.y,
                x0,
                y0,
                wh,
                src,
                y_stride,
                uni[0].0,
                uni[1].0,
                self.wp_b[2][0],
            )
        } else {
            u32::MAX
        };
        let mut bi_mv = [uni[0].0, uni[1].0];
        let (mut bi_mvd, mut bi_flag) = (best_mvd, best_flag);
        if bi_ok {
            let mut cost = bi_satd as f32 + lam * amvp_bits(2, bi_mvd, bi_flag);
            let default_wp = [0, 1, 2]
                .iter()
                .all(|&i| matches!(self.wp_b[i][0], Weighting::Default));
            if self.bi_refine > 0 && default_wp && cost < self.bi_refine_gate * best_cost {
                // The list whose own search did worse is re-searched first,
                // against the better one's prediction.
                let first = usize::from(uni[0].1 < uni[1].1);
                let planes = [&ref0.y, &ref1.y];
                let halves = [self.half_for(ref0.poc), self.half_for(ref1.poc)];
                for it in 0..2 * self.bi_refine as usize {
                    let list = (first + it) % 2;
                    let mv = self.refine_bi(
                        ctx,
                        planes,
                        halves[list].as_deref(),
                        list,
                        x0,
                        y0,
                        wh,
                        src,
                        y_stride,
                        bi_mv,
                        lam,
                        mvp[list],
                    );
                    if mv == bi_mv[list] {
                        if it > 0 {
                            break;
                        }
                        continue;
                    }
                    let mut trial = bi_mv;
                    trial[list] = mv;
                    let a = mvd_for(mv, mvp[list][0]);
                    let b = mvd_for(mv, mvp[list][1]);
                    let mut mvd = bi_mvd;
                    let mut fl = bi_flag;
                    mvd[list] = a;
                    fl[list] = 0;
                    let bits_a = amvp_bits(2, mvd, fl);
                    mvd[list] = b;
                    fl[list] = 1;
                    let bits_b = amvp_bits(2, mvd, fl);
                    if bits_a <= bits_b {
                        mvd[list] = a;
                        fl[list] = 0;
                    }
                    let satd = self.satd_bi_at(
                        ctx,
                        &ref0.y,
                        &ref1.y,
                        x0,
                        y0,
                        wh,
                        src,
                        y_stride,
                        trial[0],
                        trial[1],
                        Weighting::Default,
                    );
                    let c = satd as f32 + lam * bits_a.min(bits_b);
                    if c < cost {
                        cost = c;
                        bi_mv = trial;
                        bi_mvd = mvd;
                        bi_flag = fl;
                        bi_satd = satd;
                    } else if it > 0 {
                        break;
                    }
                }
            }
            if cost < best_cost {
                best_cost = cost;
                best = Some((2, bi_satd));
            }
        }
        let (best_idc, amvp_satd) = best.expect("three shapes were costed");

        // Merge, over the decoder's own candidates. A candidate may use
        // either list or both, and is scored the way it would be predicted.
        let mut best_merge: Option<(usize, u32)> = None;
        let mut best_merge_cost = f32::INFINITY;
        let mut seen: Vec<([Mv; 2], [i8; 2])> = Vec::with_capacity(MAX_MERGE_CAND);
        for (idx, cand) in merge.iter().enumerate().take(self.merge_tried) {
            let key = (cand.mv, cand.ref_idx);
            if cand.ref_idx == [-1, -1] || seen.contains(&key) {
                continue;
            }
            seen.push(key);
            let satd = match (cand.ref_idx[0] >= 0, cand.ref_idx[1] >= 0) {
                (true, true) => self.satd_bi_at(
                    ctx,
                    &ref0.y,
                    &ref1.y,
                    x0,
                    y0,
                    wh,
                    src,
                    y_stride,
                    cand.mv[0],
                    cand.mv[1],
                    self.wp_b[2][0],
                ),
                (true, false) => self.satd_at_weighted(
                    ctx,
                    &ref0.y,
                    x0,
                    y0,
                    wh,
                    src,
                    y_stride,
                    cand.mv[0],
                    self.wp_b[0][0],
                    0,
                ),
                (false, true) => self.satd_at_weighted(
                    ctx,
                    &ref1.y,
                    x0,
                    y0,
                    wh,
                    src,
                    y_stride,
                    cand.mv[1],
                    self.wp_b[1][0],
                    1,
                ),
                (false, false) => unreachable!("filtered above"),
            };
            let bits = if whole {
                rate.skip(idx as u8).min(rate.merge(idx as u8))
            } else {
                rate.pu(
                    &PuPick {
                        merge_idx: Some(idx as u8),
                        ..PuPick::default()
                    },
                    true,
                    pu,
                    1,
                )
            };
            let cost = satd as f32 + lam * bits;
            if cost < best_merge_cost {
                best_merge_cost = cost;
                best_merge = Some((idx, satd));
            }
        }

        match best_merge {
            Some((idx, satd)) if best_merge_cost <= best_cost => PuPick {
                merge_idx: Some(idx as u8),
                idc: 0,
                mvd: [Mv::ZERO; 2],
                mvp_flag: [0; 2],
                mv: merge[idx].mv,
                ref_idx: merge[idx].ref_idx,
                satd,
                cost: best_merge_cost,
            },
            _ => {
                // The motion the winner carries, as `ref_idx` pairs the
                // decoder would store.
                let r: [i8; 2] = match best_idc {
                    0 => [0, -1],
                    1 => [-1, 0],
                    _ => [0, 0],
                };
                let (mvd, mvp_flag, mv) = if best_idc == 2 {
                    (bi_mvd, bi_flag, bi_mv)
                } else {
                    (
                        best_mvd,
                        best_flag,
                        [
                            if r[0] >= 0 { uni[0].0 } else { Mv::ZERO },
                            if r[1] >= 0 { uni[1].0 } else { Mv::ZERO },
                        ],
                    )
                };
                PuPick {
                    merge_idx: None,
                    idc: best_idc,
                    mvd,
                    mvp_flag,
                    mv,
                    ref_idx: r,
                    satd: amvp_satd,
                    cost: best_cost,
                }
            }
        }
    }

    /// Code the CU of `1 << log2_cu` at `(x0, y0)` as one 2Nx2N prediction
    /// unit carrying `pick`: the flatness proxy, the prediction through the
    /// decoder's own MC — one list or both, weighted exactly as the slice
    /// header says those lists are, and per-format chroma for free — the
    /// residual, the kind the residual leaves (a merge with nothing left
    /// is a skip), and the motion and marks stored exactly as the
    /// decoder's `prediction_unit` stores them after parsing ("Store
    /// motion", src/hevc/ctu.rs): the next CUs' candidate lists read them —
    /// the reference index included, which is what makes a neighbour's
    /// vector scale by the right POC distance.
    #[allow(clippy::too_many_arguments)]
    fn code_whole(
        &mut self,
        ctx: &MeCtx<'_, S>,
        refs: TreeRefs<'_, S>,
        out: &mut InterCuDecision,
        pick: PuPick,
        x0: usize,
        y0: usize,
        log2_cu: u32,
        src_y: &[S],
        y_stride: usize,
        src_cb: &[S],
        src_cr: &[S],
        c_stride: usize,
    ) {
        let n = 1usize << log2_cu;
        let src = &src_y[y0 * y_stride + x0..];
        let w4 = self.info.w4;
        out.mv = pick.mv[0];
        out.ref_idx = pick.ref_idx[0];
        out.mv_l1 = pick.mv[1];
        out.ref_idx_l1 = pick.ref_idx[1];

        if self.rd_intra.is_none() && prefer_intra(ctx, pick.satd, src, y_stride, n) {
            out.kind = InterCuKind::UseIntra;
            // An intra CU's motion, stored now so later candidate
            // derivations see what the decoder will see.
            fill_motion(
                &mut self.recon.motion,
                self.recon.w4,
                x0,
                y0,
                n,
                n,
                MotionInfo::INTRA,
            );
            PicInfo::fill4(&mut self.info.pred_mode, w4, x0, y0, n, n, 1);
            // `coding_unit` records `cu_skip_flag` for every CU before it
            // knows the pred mode (ctu.rs:419), and the *next* CU's
            // `cu_skip_flag` context counts skipped neighbours out of
            // exactly this array. An intra CU is never skipped — and under
            // the quadtree the area may still hold a skip mark from a
            // trial that lost.
            PicInfo::fill4(&mut self.info.skip, w4, x0, y0, n, n, 0);
            return;
        }

        self.predict_cu(ctx, refs, &pick, x0, y0, n);

        let any = if self.no_residual {
            out.rqt_root_cbf = false;
            out.cbf_luma = false;
            out.cbf_chroma = [false; 2];
            out.cbf_chroma_bot = [false; 2];
            false
        } else {
            self.code_residual_cu(
                ctx, x0, y0, log2_cu, src, y_stride, src_cb, src_cr, c_stride, out,
            )
        };
        out.kind = match (pick.merge_idx, any, refs) {
            (Some(merge_idx), false, _) => InterCuKind::Skip { merge_idx },
            (Some(merge_idx), true, _) => InterCuKind::Merge { merge_idx },
            (None, _, TreeRefs::P(_)) => InterCuKind::Amvp {
                mvp_flag: pick.mvp_flag[0],
                mvd: pick.mvd[0],
            },
            (None, _, TreeRefs::B(..)) => InterCuKind::BAmvp {
                idc: pick.idc,
                mvd: pick.mvd,
                mvp_flag: pick.mvp_flag,
            },
        };

        let mi = self.motion_info(&pick, refs);
        fill_motion(&mut self.recon.motion, self.recon.w4, x0, y0, n, n, mi);
        PicInfo::fill4(&mut self.info.pred_mode, w4, x0, y0, n, n, 0);
        PicInfo::fill4(
            &mut self.info.skip,
            w4,
            x0,
            y0,
            n,
            n,
            matches!(out.kind, InterCuKind::Skip { .. }) as u8,
        );
    }

    /// The cheapest partitioned alternative for the CU of `1 << log2_cu` at
    /// `(x0, y0)`, among the shapes [`Self::parts`] offers, or `None` when it
    /// offers none. Each shape's two units are picked in the reader's
    /// order, and before the second unit's candidates are derived the CU
    /// is marked inter and the first unit's motion is in the motion grid —
    /// what `coding_unit` and `prediction_unit` have left there for the
    /// reader by then, and what makes the second unit's candidate list the
    /// one the reader builds (the first unit is a neighbour of the second,
    /// and the standard's second-unit exclusions depend on it). The grid
    /// and the marks are put back afterwards; the caller stores the
    /// winner's. A shape's cost is its two units' ([`PuPick::cost`]) plus
    /// the CU syntax around them ([`Rate::parts`]), comparable with a
    /// 2Nx2N unit's cost, which prices the whole CU's syntax.
    #[allow(clippy::too_many_arguments)]
    fn pick_parts(
        &mut self,
        ctx: &MeCtx<'_, S>,
        refs: TreeRefs<'_, S>,
        x0: usize,
        y0: usize,
        log2_cu: u32,
        src_y: &[S],
        y_stride: usize,
        rate: &Rate,
        lam: f32,
    ) -> Option<PartsPick> {
        let shapes: &[PartMode] = match self.parts {
            InterParts::None => return None,
            InterParts::Symmetric => &[PartMode::P2NxN, PartMode::PNx2N],
        };
        let n = 1usize << log2_cu;
        let w4 = self.info.w4;
        let motion = save4(&self.recon.motion, self.recon.w4, x0, y0, n);
        let pred_mode = save4(&self.info.pred_mode, w4, x0, y0, n);
        PicInfo::fill4(&mut self.info.pred_mode, w4, x0, y0, n, n, 0);
        let mut best: Option<PartsPick> = None;
        for &part in shapes {
            let mut picks = [PuPick::default(); 2];
            let mut cost = lam * rate.parts(part);
            for (i, &(px, py, pw, ph)) in part.pus(n as i32).iter().enumerate() {
                let pu = PuPos {
                    x_cb: x0 as i32,
                    y_cb: y0 as i32,
                    n_cb: n as i32,
                    x_pb: x0 as i32 + px,
                    y_pb: y0 as i32 + py,
                    w: pw,
                    h: ph,
                    part_idx: i as u32,
                };
                let pick = match refs {
                    TreeRefs::P(l0) => self.pick_pu(ctx, l0, &pu, src_y, y_stride, rate, lam),
                    TreeRefs::B(r0, r1) => {
                        self.pick_pu_b(ctx, r0, r1, &pu, src_y, y_stride, rate, lam)
                    }
                };
                cost += pick.cost;
                picks[i] = pick;
                if i == 0 {
                    let mi = self.motion_info(&pick, refs);
                    fill_motion(
                        &mut self.recon.motion,
                        self.recon.w4,
                        pu.x_pb as usize,
                        pu.y_pb as usize,
                        pw as usize,
                        ph as usize,
                        mi,
                    );
                }
            }
            restore4(&mut self.recon.motion, self.recon.w4, x0, y0, n, &motion);
            if best.as_ref().is_none_or(|b| cost < b.2) {
                best = Some((part, picks, cost));
            }
        }
        restore4(&mut self.info.pred_mode, w4, x0, y0, n, &pred_mode);
        best
    }

    /// The motion `prediction_unit` stores for a unit carrying `pick`: its
    /// vectors and reference indices, and the POC distance of each list it
    /// uses — which is what makes a neighbour's vector scale right in a
    /// later unit's candidate derivation.
    fn motion_info(&self, pick: &PuPick, refs: TreeRefs<'_, S>) -> MotionInfo {
        let mut mi = MotionInfo {
            mv: pick.mv,
            ref_delta: [0; 2],
            ref_idx: pick.ref_idx,
            flags: 0,
            pad: 0,
        };
        for list in 0..2usize {
            if pick.ref_idx[list] >= 0 {
                let poc = match refs {
                    TreeRefs::P(l0) => l0[pick.ref_idx[0] as usize].poc,
                    TreeRefs::B(r0, r1) => {
                        if list == 0 {
                            r0.poc
                        } else {
                            r1.poc
                        }
                    }
                };
                mi.ref_delta[list] =
                    (self.cur_poc - poc).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            }
        }
        mi
    }

    /// Code the CU of `1 << log2_cu` at `(x0, y0)` as the partitioned shape
    /// `part` whose two units carry `pus` ([`Self::pick_parts`]'s winner):
    /// the flatness proxy against the units' summed SATD, then each unit
    /// predicted through the decoder's own MC, one residual over the whole
    /// CU (one CU-sized transform unit — the SPS's
    /// `max_transform_hierarchy_depth_inter` of 2 infers no split), and the
    /// motion and marks stored per unit as `prediction_unit` stores them.
    #[allow(clippy::too_many_arguments)]
    fn code_parts(
        &mut self,
        ctx: &MeCtx<'_, S>,
        refs: TreeRefs<'_, S>,
        out: &mut InterCuDecision,
        part: PartMode,
        pus: [PuPick; 2],
        x0: usize,
        y0: usize,
        log2_cu: u32,
        src_y: &[S],
        y_stride: usize,
        src_cb: &[S],
        src_cr: &[S],
        c_stride: usize,
    ) {
        let n = 1usize << log2_cu;
        let src = &src_y[y0 * y_stride + x0..];
        let w4 = self.info.w4;
        out.part = part;
        out.pus = pus;
        out.mv = pus[0].mv[0];
        out.ref_idx = pus[0].ref_idx[0];
        out.mv_l1 = pus[0].mv[1];
        out.ref_idx_l1 = pus[0].ref_idx[1];
        if prefer_intra(
            ctx,
            pus[0].satd.saturating_add(pus[1].satd),
            src,
            y_stride,
            n,
        ) {
            out.kind = InterCuKind::UseIntra;
            out.part = PartMode::P2Nx2N;
            fill_motion(
                &mut self.recon.motion,
                self.recon.w4,
                x0,
                y0,
                n,
                n,
                MotionInfo::INTRA,
            );
            PicInfo::fill4(&mut self.info.pred_mode, w4, x0, y0, n, n, 1);
            PicInfo::fill4(&mut self.info.skip, w4, x0, y0, n, n, 0);
            return;
        }
        let rects = part.pus(n as i32);
        for (pk, &(px, py, pw, ph)) in pus.iter().zip(rects.iter()) {
            let (x, y) = (x0 + px as usize, y0 + py as usize);
            match refs {
                TreeRefs::P(l0) => {
                    let r = pk.ref_idx[0] as usize;
                    let wp = self.wp_for(r);
                    predict_block(
                        ctx.dsp,
                        &mut self.scratch,
                        &mut self.recon,
                        x,
                        y,
                        pw as usize,
                        ph as usize,
                        Some((l0[r], pk.mv[0])),
                        None,
                        wp,
                    );
                }
                TreeRefs::B(r0, r1) => {
                    let a = (pk.ref_idx[0] >= 0).then_some((r0, pk.mv[0]));
                    let b = (pk.ref_idx[1] >= 0).then_some((r1, pk.mv[1]));
                    let wp = self.wp_b_for(pk.ref_idx);
                    predict_block(
                        ctx.dsp,
                        &mut self.scratch,
                        &mut self.recon,
                        x,
                        y,
                        pw as usize,
                        ph as usize,
                        a,
                        b,
                        wp,
                    );
                }
            }
        }
        self.code_residual_cu(
            ctx, x0, y0, log2_cu, src, y_stride, src_cb, src_cr, c_stride, out,
        );
        out.kind = InterCuKind::Parts;
        for (pk, &(px, py, pw, ph)) in pus.iter().zip(rects.iter()) {
            let mi = self.motion_info(pk, refs);
            fill_motion(
                &mut self.recon.motion,
                self.recon.w4,
                x0 + px as usize,
                y0 + py as usize,
                pw as usize,
                ph as usize,
                mi,
            );
        }
        PicInfo::fill4(&mut self.info.pred_mode, w4, x0, y0, n, n, 0);
        PicInfo::fill4(&mut self.info.skip, w4, x0, y0, n, n, 0);
    }

    /// The residual of one 2Nx2N CU, luma then every chroma TB this
    /// format carries, each reconstructed in place through the decoder's
    /// inverse path. Fills `cbf_luma`, `cbf_chroma`, `cbf_chroma_bot`,
    /// `rqt_root_cbf` and the coefficient arrays; returns `rqt_root_cbf`.
    ///
    /// Shared by the P and B walks precisely so the two cannot drift: the
    /// residual does not depend on how the prediction was signalled, only
    /// on what the prediction left behind, and a second copy of the
    /// chroma-format placement is the drift hazard this crate keeps
    /// finding. `predict_block` must already have written the prediction
    /// into `recon`.
    #[allow(clippy::too_many_arguments)]
    fn code_residual_cu(
        &mut self,
        ctx: &MeCtx<'_, S>,
        x0: usize,
        y0: usize,
        log2_cu: u32,
        src: &[S],
        y_stride: usize,
        src_cb: &[S],
        src_cr: &[S],
        c_stride: usize,
        out: &mut InterCuDecision,
    ) -> bool {
        out.split_tu = false;
        self.tree_bits = None;
        let trial = self.tu_split
            && !ctx.bypass
            && (3..=self.tu_split_max.min(5)).contains(&log2_cu)
            && self.cat <= 1;
        if !trial {
            return self.code_residual_whole(
                ctx, x0, y0, log2_cu, src, y_stride, src_cb, src_cr, c_stride, out,
            );
        }
        let n = 1usize << log2_cu;
        // At 8x8 the split leaves 4:2:0 chroma whole (4x4 children carry
        // none; the fourth codes the parent's), so only luma is retried.
        let rcat = if log2_cu == 3 { 0 } else { self.cat };
        let mut save = std::mem::take(&mut self.tu_save);
        Self::save_region(&self.recon, rcat, x0, y0, n, &mut save);
        let any = self.code_residual_whole(
            ctx, x0, y0, log2_cu, src, y_stride, src_cb, src_cr, c_stride, out,
        );
        if !out.cbf_luma
            || (self.tu_split_nz > 1
                && out.luma[..n * n].iter().filter(|&&v| v != 0).count() < self.tu_split_nz)
        {
            self.tu_save = save;
            return any;
        }
        // The CU's SSD, as `cu_ssd` weighs it, over the luma source this
        // function is handed already at the CU (`src`) and the chroma
        // planes it is handed whole.
        let cat = self.cat;
        let cu_err = |recon: &Frame<S>| -> u64 {
            let yo = recon.y.offset(x0 as isize, y0 as isize);
            let mut e = (ctx.dist.ssd)(src, y_stride, &recon.y.data[yo..], recon.y.stride, n, n);
            if cat != 0 {
                let (cx, cy, nc) = (x0 / 2, y0 / 2, n / 2);
                let mut c = 0u64;
                for (p, sp) in [(&recon.cb, src_cb), (&recon.cr, src_cr)] {
                    let o = p.offset(cx as isize, cy as isize);
                    c += (ctx.dist.ssd)(
                        &sp[cy * c_stride + cx..],
                        c_stride,
                        &p.data[o..],
                        p.stride,
                        nc,
                        nc,
                    );
                }
                let bd_off = 6 * (ctx.bit_depth as i32 - 8);
                let qp_c = chroma_qp(cat, ctx.qp.clamp(-bd_off, 57));
                e += (c as f64 * 2f64.powf(f64::from(ctx.qp - qp_c) / 3.0)).round() as u64;
            }
            e
        };
        let lam = ssd_lambda(ctx.qp, ctx.bit_depth);
        let is_b = self.slice_b;
        let tree_whole = crate::encode::h265::inter_tree_bits(out, self.cat, is_b);
        let j_whole = cu_err(&self.recon) as f64 + lam * f64::from(tree_whole as f32);
        let mut keep = std::mem::take(&mut self.tu_keep);
        Self::save_region(&self.recon, rcat, x0, y0, n, &mut keep);
        Self::restore_region(&mut self.recon, rcat, x0, y0, n, &save);
        let mut alt = std::mem::take(&mut self.tu_alt);
        alt.log2_cu = log2_cu;
        alt.bypass = false;
        alt.sign_hiding = out.sign_hiding;
        alt.split_tu = true;
        if log2_cu == 3 {
            alt.cbf_chroma = out.cbf_chroma;
            for c in 0..2 {
                alt.chroma[c][..16].copy_from_slice(&out.chroma[c][..16]);
            }
        }
        let any_split = self.code_residual_split(
            ctx, x0, y0, log2_cu, src, y_stride, src_cb, src_cr, c_stride, &mut alt,
        );
        let tree_split = if any_split {
            crate::encode::h265::inter_tree_bits(&alt, self.cat, is_b)
        } else {
            0.0
        };
        let j_split = if any_split {
            cu_err(&self.recon) as f64 + lam * f64::from(tree_split as f32)
        } else {
            f64::INFINITY
        };
        let any = if j_split < j_whole {
            let q = n * n;
            out.luma[..q].copy_from_slice(&alt.luma[..q]);
            if log2_cu > 3 {
                let qc = q / 4;
                for c in 0..2 {
                    out.chroma[c][..qc].copy_from_slice(&alt.chroma[c][..qc]);
                }
            }
            out.split_tu = true;
            out.cbf_luma = alt.cbf_luma;
            out.cbf_chroma = alt.cbf_chroma;
            out.cbf_chroma_bot = [false; 2];
            out.cbf_luma_tu = alt.cbf_luma_tu;
            out.cbf_chroma_tu = alt.cbf_chroma_tu;
            out.rqt_root_cbf = true;
            if alt.qp_y == out.qp_y {
                self.tree_bits = Some(tree_split);
            }
            true
        } else {
            Self::restore_region(&mut self.recon, rcat, x0, y0, n, &keep);
            self.tree_bits = Some(tree_whole);
            any
        };
        self.tu_save = save;
        self.tu_keep = keep;
        self.tu_alt = alt;
        any
    }

    /// Copy the luma and (4:2:0) chroma samples of the `n`-square CU at
    /// `(x0, y0)` out of `recon` into `buf`.
    fn save_region(recon: &Frame<S>, cat: u32, x0: usize, y0: usize, n: usize, buf: &mut Vec<S>) {
        let nc = if cat == 0 { 0 } else { n / 2 };
        buf.clear();
        let mut one = |p: &Plane16<S>, x: usize, y: usize, m: usize| {
            for r in 0..m {
                let o = p.offset(x as isize, (y + r) as isize);
                buf.extend_from_slice(&p.data[o..o + m]);
            }
        };
        one(&recon.y, x0, y0, n);
        if nc > 0 {
            one(&recon.cb, x0 / 2, y0 / 2, nc);
            one(&recon.cr, x0 / 2, y0 / 2, nc);
        }
    }

    /// Put back what [`Self::save_region`] took.
    fn restore_region(recon: &mut Frame<S>, cat: u32, x0: usize, y0: usize, n: usize, buf: &[S]) {
        let nc = if cat == 0 { 0 } else { n / 2 };
        let mut at = 0;
        let mut one = |p: &mut Plane16<S>, x: usize, y: usize, m: usize| {
            for r in 0..m {
                let o = p.offset(x as isize, (y + r) as isize);
                p.data[o..o + m].copy_from_slice(&buf[at..at + m]);
                at += m;
            }
        };
        one(&mut recon.y, x0, y0, n);
        if nc > 0 {
            one(&mut recon.cb, x0 / 2, y0 / 2, nc);
            one(&mut recon.cr, x0 / 2, y0 / 2, nc);
        }
    }

    /// The split shape of [`Self::code_residual_cu`]: four quarter TUs of
    /// luma and, at 4:2:0, four quarter TBs per chroma component, each
    /// coded and reconstructed in place over the prediction. Fills the
    /// split fields of `out`; returns whether anything was coded.
    #[allow(clippy::too_many_arguments)]
    fn code_residual_split(
        &mut self,
        ctx: &MeCtx<'_, S>,
        x0: usize,
        y0: usize,
        log2_cu: u32,
        src: &[S],
        y_stride: usize,
        src_cb: &[S],
        src_cr: &[S],
        c_stride: usize,
        out: &mut InterCuDecision,
    ) -> bool {
        let qp_l = ctx.qp + 6 * (ctx.bit_depth as i32 - 8);
        let h = 1usize << (log2_cu - 1);
        let q = h * h;
        let mut any = false;
        for i in 0..4 {
            let (x, y) = (x0 + (i & 1) * h, y0 + (i >> 1) * h);
            let nz = code_residual_inter(
                ctx,
                &mut self.recon.y,
                x,
                y,
                log2_cu - 1,
                0,
                qp_l,
                &src[(y - y0) * y_stride + (x - x0)..],
                y_stride,
                &mut out.luma[i * q..(i + 1) * q],
            );
            out.cbf_luma_tu[i] = nz != 0;
            any |= nz != 0;
        }
        out.cbf_luma = out.cbf_luma_tu.iter().any(|&c| c);
        out.cbf_chroma_tu = [[false; 4]; 2];
        if log2_cu == 3 {
            // The chroma TB stays whole; the caller put the unsplit
            // shape's levels and cbfs in place.
            out.cbf_chroma_bot = [false; 2];
            out.rqt_root_cbf = any || out.cbf_chroma.iter().any(|&c| c);
            return any;
        }
        let cctx = &self.chroma_ctx(ctx);
        if self.cat != 0 {
            let bd_off = 6 * (ctx.bit_depth as i32 - 8);
            let qp_c = chroma_qp(self.cat, ctx.qp.clamp(-bd_off, 57)) + bd_off;
            let hc = h / 2;
            let qc = hc * hc;
            for comp in 0..2usize {
                let plane = if comp == 0 {
                    &mut self.recon.cb
                } else {
                    &mut self.recon.cr
                };
                let srcp = if comp == 0 { src_cb } else { src_cr };
                for i in 0..4 {
                    let (px, py) = (x0 / 2 + (i & 1) * hc, y0 / 2 + (i >> 1) * hc);
                    let nz = code_residual_inter(
                        cctx,
                        plane,
                        px,
                        py,
                        log2_cu - 2,
                        comp + 1,
                        qp_c,
                        &srcp[py * c_stride + px..],
                        c_stride,
                        &mut out.chroma[comp][i * qc..(i + 1) * qc],
                    );
                    out.cbf_chroma_tu[comp][i] = nz != 0;
                    any |= nz != 0;
                }
            }
        }
        out.cbf_chroma = [
            out.cbf_chroma_tu[0].iter().any(|&c| c),
            out.cbf_chroma_tu[1].iter().any(|&c| c),
        ];
        out.cbf_chroma_bot = [false; 2];
        out.rqt_root_cbf = any;
        any
    }

    /// The unsplit residual of [`Self::code_residual_cu`].
    #[allow(clippy::too_many_arguments)]
    fn code_residual_whole(
        &mut self,
        ctx: &MeCtx<'_, S>,
        x0: usize,
        y0: usize,
        log2_cu: u32,
        src: &[S],
        y_stride: usize,
        src_cb: &[S],
        src_cr: &[S],
        c_stride: usize,
        out: &mut InterCuDecision,
    ) -> bool {
        let qp_l = ctx.qp + 6 * (ctx.bit_depth as i32 - 8);
        let nz_l = code_residual_inter(
            ctx,
            &mut self.recon.y,
            x0,
            y0,
            log2_cu,
            0,
            qp_l,
            src,
            y_stride,
            &mut out.luma,
        );
        let mut nz_c = 0u32;
        let cctx = &self.chroma_ctx(ctx);
        if self.cat != 0 {
            // QpC: Table 8-10 for 4:2:0, `Min(qPi, 51)` otherwise — the
            // decoder's own `hevc::ctu::chroma_qp`, told which format it is
            // rather than the constant 1 this module used while it modelled
            // 4:2:0 alone. The PPS and slice header write zero cb/cr
            // offsets, so `qPi` is the luma QP.
            let bd_off = 6 * (ctx.bit_depth as i32 - 8);
            let qp_c = chroma_qp(self.cat, ctx.qp.clamp(-bd_off, 57)) + bd_off;
            let (sw, sh) = sub_wh(self.cat);
            // Where this CU's chroma TBs sit and how big they are, from the
            // placement `transform_unit` performs — `here` plus its 4:2:2
            // `yct = yc + t * nc` pair. Anchors come back in *luma*
            // coordinates; dividing by (SubWidthC, SubHeightC) puts them on
            // the chroma plane, which is also how the source is addressed.
            let (tbs, ntb, log2c) = chroma_tbs(self.cat, x0, y0, log2_cu);
            let nc2 = 1usize << (2 * log2c);
            for comp in 0..2usize {
                let plane = if comp == 0 {
                    &mut self.recon.cb
                } else {
                    &mut self.recon.cr
                };
                let srcp = if comp == 0 { src_cb } else { src_cr };
                for (t, &(ax, ay)) in tbs[..ntb].iter().enumerate() {
                    let (px, py) = (ax / sw, ay / sh);
                    let soff = py * c_stride + px;
                    let levels = &mut out.chroma[comp][t * nc2..(t + 1) * nc2];
                    let nz = code_residual_inter(
                        cctx,
                        plane,
                        px,
                        py,
                        log2c,
                        comp + 1,
                        qp_c,
                        &srcp[soff..],
                        c_stride,
                        levels,
                    );
                    if t == 0 {
                        out.cbf_chroma[comp] = nz != 0;
                    } else {
                        out.cbf_chroma_bot[comp] = nz != 0;
                    }
                    nz_c += nz;
                }
            }
        }
        out.cbf_luma = nz_l != 0;
        out.rqt_root_cbf = nz_l + nz_c != 0;
        out.rqt_root_cbf
    }

    /// Code the CTU at `(cu_x, cu_y)` as an **intra** CU inside this P
    /// slice, after [`InterPicture::code_ctu`] answered
    /// [`InterCuKind::UseIntra`] for it. Call it only then, and only
    /// immediately: the walk's ordering invariants are the intra
    /// decision's too.
    ///
    /// This is not a second intra encoder. It is
    /// `code_cu_2nx2n_intra` — the very function
    /// [`super::h265_intra::IntraPicture::code_ctu`] calls for an I
    /// slice — pointed at *this* picture's state:
    ///
    /// - **`self.recon`**, so intra prediction reads reconstructed
    ///   neighbours *including the inter-coded ones*. That is legal and
    ///   deliberate rather than an oversight of constrained intra
    ///   prediction: `write_pps` writes `constrained_intra_pred_flag` 0
    ///   (`h265_syntax.rs:272`), which switches off the second half of
    ///   the reader's own reference check —
    ///   `available_at(..) && (!cip || pred_mode == 1)`, `ctu.rs:1157`.
    ///   With `cip` false the reader takes any decoded neighbour, so the
    ///   encoder must too, or the two predict from different samples.
    /// - **`self.info.intra_mode`**, the decoder's own per-4x4 luma-mode
    ///   grid, as the grid the MPM derivation reads and fills — rather
    ///   than a private copy that would have to be kept in step with it.
    /// - **`self.info.pred_mode`**, so the MPM derivation applies the
    ///   reader's not-intra gate (`ctu.rs:627`): a neighbouring *inter*
    ///   CU contributes `INTRA_DC`, not whatever mode last stood in the
    ///   mode grid at that position.
    ///
    /// `code_ctu` has already stored the motion (`MotionInfo::INTRA`),
    /// `pred_mode` 1 and `skip` 0 over the CU — the marks
    /// `coding_unit` writes before it parses any intra syntax — so the
    /// deblocker and every later candidate derivation see what a decoder
    /// of this stream will see.
    #[allow(clippy::too_many_arguments)]
    pub fn code_ctu_intra(
        &mut self,
        ctx: &MeCtx<'_, S>,
        cu_x: usize,
        cu_y: usize,
        src_y: &[S],
        y_stride: usize,
        src_cb: &[S],
        src_cr: &[S],
        c_stride: usize,
    ) -> CuDecision {
        let n = 1usize << self.log2_ctb;
        self.code_cu_intra(
            ctx,
            cu_x * n,
            cu_y * n,
            self.log2_ctb,
            src_y,
            y_stride,
            src_cb,
            src_cr,
            c_stride,
        )
    }

    /// [`Self::code_ctu_intra`] for one coding unit of `1 << log2_cu` at
    /// luma `(x0, y0)`: the CU [`Self::code_cu`] or [`Self::code_cu_b`]
    /// answered [`InterCuKind::UseIntra`] for, called immediately after.
    #[allow(clippy::too_many_arguments)]
    pub fn code_cu_intra(
        &mut self,
        ctx: &MeCtx<'_, S>,
        x0: usize,
        y0: usize,
        log2_cu: u32,
        src_y: &[S],
        y_stride: usize,
        src_cb: &[S],
        src_cr: &[S],
        c_stride: usize,
    ) -> CuDecision {
        // Split the borrows: the mode grid is written, the pred-mode grid
        // is read, and both live in `info` beside each other.
        let PicInfo {
            intra_mode,
            pred_mode,
            ..
        } = &mut self.info;
        let (intra_mode, pred_mode) = (&mut intra_mode[..], &pred_mode[..]);
        code_cu_2nx2n_intra(
            ctx,
            self.geo,
            &mut self.recon,
            intra_mode,
            Some(pred_mode),
            &mut self.intra_scratch,
            self.split_depth,
            x0,
            y0,
            log2_cu,
            src_y,
            y_stride,
            src_cb,
            src_cr,
            c_stride,
        )
    }

    /// Decide and code the CTB at `(cu_x, cu_y)` (CTB units) of a P or B
    /// picture as a coding quadtree — the walk and the cost of
    /// [`super::h265_intra::IntraPicture::code_ctu_tree`] over this
    /// picture's units: every node coded whole by the inter decision
    /// ([`Self::code_cu`] or [`Self::code_cu_b`], handing over to
    /// [`Self::code_cu_intra`] where it answers [`InterCuKind::UseIntra`])
    /// and then split, the cheaper kept.
    ///
    /// A trial here leaves more behind than an intra one: beside the
    /// samples, the motion grid and the `pred_mode`, `skip` and
    /// `intra_mode` marks the candidate derivations read. All of it is put
    /// back when the whole node wins (`TrialSave`), over exactly the
    /// node's area — and nothing outside it is ever written — so the next
    /// unit's merge and AMVP lists are built from the winners alone, as a
    /// decoder builds them from what it parsed.
    #[allow(clippy::too_many_arguments)]
    pub fn code_ctu_tree(
        &mut self,
        ctx: &MeCtx<'_, S>,
        want: &dyn Fn(usize, usize, u32) -> i32,
        max_depth: u32,
        refs: TreeRefs<'_, S>,
        cu_x: usize,
        cu_y: usize,
        src: &Srcs<'_, S>,
    ) -> Vec<TreeCu<PCuDecision>> {
        let log2 = self.log2_ctb;
        let mut out = Vec::new();
        self.tree_node(
            ctx,
            want,
            max_depth,
            refs,
            cu_x << log2,
            cu_y << log2,
            log2,
            0,
            src,
            &mut out,
        );
        out
    }

    /// Make the inter prediction of `pick` over the `n`-square unit at
    /// `(x0, y0)` in the reconstruction.
    fn predict_cu(
        &mut self,
        ctx: &MeCtx<'_, S>,
        refs: TreeRefs<'_, S>,
        pick: &PuPick,
        x0: usize,
        y0: usize,
        n: usize,
    ) {
        match refs {
            TreeRefs::P(l0) => {
                let r = pick.ref_idx[0] as usize;
                let wp = self.wp_for(r);
                predict_block(
                    ctx.dsp,
                    &mut self.scratch,
                    &mut self.recon,
                    x0,
                    y0,
                    n,
                    n,
                    Some((l0[r], pick.mv[0])),
                    None,
                    wp,
                );
            }
            TreeRefs::B(r0, r1) => {
                let a = (pick.ref_idx[0] >= 0).then_some((r0, pick.mv[0]));
                let b = (pick.ref_idx[1] >= 0).then_some((r1, pick.mv[1]));
                let wp = self.wp_b_for(pick.ref_idx);
                predict_block(
                    ctx.dsp,
                    &mut self.scratch,
                    &mut self.recon,
                    x0,
                    y0,
                    n,
                    n,
                    a,
                    b,
                    wp,
                );
            }
        }
    }

    /// The context an inter unit's chroma is coded under: `ctx`, without
    /// RDOQ unless [`Self::rdoq_chroma`].
    fn chroma_ctx<'a>(&self, ctx: &MeCtx<'a, S>) -> MeCtx<'a, S> {
        IntraCtx {
            rdoq: ctx.rdoq.filter(|_| self.rdoq_chroma),
            ..*ctx
        }
    }

    /// A decision to decide into: one the walk discarded, or a new one.
    fn fresh(&mut self, log2_cu: u32, ctx: &MeCtx<'_, S>) -> Box<InterCuDecision> {
        let mut d = self.pool.pop().unwrap_or_default();
        d.reset(log2_cu, ctx);
        d
    }

    /// What a trial over the `n`-square region at `(x0, y0)` may write, as
    /// it stands, in buffers an earlier trial has finished with.
    fn take_save(&mut self, x0: usize, y0: usize, n: usize) -> TrialSave<S> {
        let mut t = self.saves.pop().unwrap_or_else(TrialSave::empty);
        t.fill(self, x0, y0, n);
        t
    }

    /// Put back what [`Self::take_save`] copied, and keep its buffers.
    fn restore(&mut self, t: TrialSave<S>, x0: usize, y0: usize, n: usize) {
        t.put(self, x0, y0, n);
        self.saves.push(t);
    }

    /// Keep a discarded decision for [`Self::fresh`].
    fn recycle(&mut self, d: PCuDecision) {
        if let PCuDecision::Inter(b) = d {
            self.pool.push(b);
        }
    }

    /// One node of [`Self::code_ctu_tree`]: returns its cost, having
    /// pushed its winning units onto `out`.
    #[allow(clippy::too_many_arguments)]
    fn tree_node(
        &mut self,
        ctx: &MeCtx<'_, S>,
        want: &dyn Fn(usize, usize, u32) -> i32,
        max_depth: u32,
        refs: TreeRefs<'_, S>,
        x0: usize,
        y0: usize,
        log2: u32,
        depth: u32,
        src: &Srcs<'_, S>,
        out: &mut Vec<TreeCu<PCuDecision>>,
    ) -> f64 {
        // A node crossing the picture edge: the split the reader infers,
        // over the children inside the picture — as in an I picture.
        let size = 1usize << log2;
        if x0 + size > self.geo.width || y0 + size > self.geo.height {
            let half = size / 2;
            let mut j = 0.0;
            for i in 0..4 {
                let (x, y) = (x0 + (i & 1) * half, y0 + (i >> 1) * half);
                if x < self.geo.width && y < self.geo.height {
                    j += self.tree_node(
                        ctx,
                        want,
                        max_depth,
                        refs,
                        x,
                        y,
                        log2 - 1,
                        depth + 1,
                        src,
                        out,
                    );
                }
            }
            return j;
        }
        let qp = want(x0, y0, log2);
        let cctx = IntraCtx { qp, ..*ctx };
        let init = if matches!(refs, TreeRefs::B(..)) {
            2
        } else {
            1
        };
        let lam = ssd_lambda(qp, ctx.bit_depth);
        self.gate_now = depth < max_depth && log2 > MIN_CB_LOG2.max(self.split_gate_above);
        self.gated = false;
        let (d, ssd, unit_bits) = self.tree_leaf(&cctx, refs, x0, y0, log2, depth, src);
        self.gate_now = false;
        let gated = std::mem::take(&mut self.gated);
        if (depth as usize) < 4 {
            self.parent_ref[depth as usize] = match &d {
                PCuDecision::Inter(i) => Some((x0 as u32, y0 as u32, i.ref_idx)),
                PCuDecision::Intra(_) => None,
            };
        }
        let flag = if log2 > MIN_CB_LOG2 {
            crate::encode::h265::split_flag_bits(init, qp, false)
        } else {
            0.0
        };
        let bits = unit_bits + flag;
        let j_whole = if gated {
            f64::INFINITY
        } else {
            ssd as f64 + lam * f64::from(bits)
        };
        // A skipped unit this cheap is not split further: see `rskip`.
        let rskip = self
            .rskip
            .is_some_and(|t| j_whole < f64::from(t) * lam * (size * size) as f64)
            && matches!(&d, PCuDecision::Inter(i) if matches!(i.kind, InterCuKind::Skip { .. }));
        if depth >= max_depth || log2 <= MIN_CB_LOG2 || rskip {
            out.push(TreeCu {
                x0,
                y0,
                log2,
                depth,
                bits,
                d,
            });
            return j_whole;
        }
        let n = 1usize << log2;
        let saved = self.take_save(x0, y0, n);
        let mark = out.len();
        let mut j_split = lam * f64::from(crate::encode::h265::split_flag_bits(init, qp, true));
        let half = n / 2;
        for i in 0..4 {
            if j_split > j_whole {
                break;
            }
            j_split += self.tree_node(
                ctx,
                want,
                max_depth,
                refs,
                x0 + (i & 1) * half,
                y0 + (i >> 1) * half,
                log2 - 1,
                depth + 1,
                src,
                out,
            );
        }
        if j_whole <= j_split {
            self.restore(saved, x0, y0, n);
            while out.len() > mark {
                let cu = out.pop().expect("above the mark");
                self.recycle(cu.d);
            }
            out.push(TreeCu {
                x0,
                y0,
                log2,
                depth,
                bits,
                d,
            });
            j_whole
        } else {
            self.recycle(d);
            self.saves.push(saved);
            j_split
        }
    }

    /// Code the unit of `1 << log2` at `(x0, y0)` whole, as a quadtree
    /// leaf: the decision (inter, or the intra one it handed over to), its
    /// reconstruction SSD, and its counted syntax bits without the
    /// `split_cu_flag`.
    ///
    /// The inter decision ([`Self::tree_leaf_coded`]), then the same unit
    /// without its residual where that is cheaper
    /// ([`Self::tree_leaf_drop`]), then — under [`Self::rd_intra`] — the
    /// unit coded intra, kept where its SSD plus lambda times its bits is
    /// lower. See `rd_intra` for which units try.
    #[allow(clippy::too_many_arguments)]
    fn tree_leaf(
        &mut self,
        ctx: &MeCtx<'_, S>,
        refs: TreeRefs<'_, S>,
        x0: usize,
        y0: usize,
        log2: u32,
        depth: u32,
        src: &Srcs<'_, S>,
    ) -> (PCuDecision, u64, f32) {
        let (coded, ssd, bits) = self.tree_leaf_coded(ctx, refs, x0, y0, log2, depth, src);
        if self.gated {
            return (coded, ssd, bits);
        }
        let (coded, ssd, bits) = if self.drop_trial {
            self.tree_leaf_drop(ctx, refs, x0, y0, log2, depth, src, coded, ssd, bits)
        } else {
            (coded, ssd, bits)
        };
        let Some(gate) = self.rd_intra else {
            return (coded, ssd, bits);
        };
        let PCuDecision::Inter(d) = &coded else {
            return (coded, ssd, bits);
        };
        let n = 1usize << log2;
        let lam = ssd_lambda(ctx.qp, ctx.bit_depth);
        let j = ssd as f64 + lam * f64::from(bits);
        if ctx.bypass
            || matches!(d.kind, InterCuKind::Skip { .. })
            || j < f64::from(gate) * lam * (n * n) as f64
        {
            return (coded, ssd, bits);
        }
        let (nref, is_b) = match refs {
            TreeRefs::P(l0) => (l0.len() as u32, false),
            TreeRefs::B(..) => (1, true),
        };
        let saved = self.take_save(x0, y0, n);
        // A cheap look first: the best intra mode's SATD cost against the
        // inter unit's search cost, both in the SATD domain.
        if let Some(g) = self.intra_satd_gate {
            let InterPicture {
                recon,
                info,
                intra_scratch,
                geo,
                ..
            } = self;
            let est = super::h265_intra::intra_estimate(
                ctx,
                *geo,
                recon,
                &info.intra_mode,
                Some(&info.pred_mode),
                intra_scratch,
                x0,
                y0,
                log2,
                src.y,
                src.y_stride,
            );
            if est > g * self.last_pick_cost {
                self.restore(saved, x0, y0, n);
                return (coded, ssd, bits);
            }
        }
        // The unit's grids say intra from here on: the intra decision's
        // own neighbours are outside it, and what it leaves must be what a
        // decoder of an intra unit holds.
        let w4 = self.info.w4;
        PicInfo::fill4(&mut self.info.pred_mode, w4, x0, y0, n, n, 1);
        PicInfo::fill4(&mut self.info.skip, w4, x0, y0, n, n, 0);
        fill_motion(
            &mut self.recon.motion,
            self.recon.w4,
            x0,
            y0,
            n,
            n,
            MotionInfo::INTRA,
        );
        let intra = PCuDecision::Intra(Box::new(self.code_cu_intra(
            ctx,
            x0,
            y0,
            log2,
            src.y,
            src.y_stride,
            src.cb,
            src.cr,
            src.c_stride,
        )));
        let ssd_i = cu_ssd(ctx, &self.recon, self.cat, x0, y0, n, src);
        let bits_i =
            crate::encode::h265::p_cu_bits(&intra, self.cat, ctx.qp, ctx.bypass, is_b, nref, depth);
        let j_i = (ssd_i as f64) + lam * f64::from(bits_i);
        // At the minimum coding block the intra unit may also take
        // `PART_NxN`, four 4x4 luma blocks with a mode each; tried where
        // the 2Nx2N intra unit came within a quarter of the inter one.
        if log2 == MIN_CB_LOG2 && j_i < self.nxn_trial * j {
            let mid = self.take_save(x0, y0, n);
            let nxn = PCuDecision::Intra(Box::new(self.code_cu_intra_nxn(
                ctx,
                x0,
                y0,
                src.y,
                src.y_stride,
                src.cb,
                src.cr,
                src.c_stride,
            )));
            let ssd_n = cu_ssd(ctx, &self.recon, self.cat, x0, y0, n, src);
            let bits_n = crate::encode::h265::p_cu_bits(
                &nxn, self.cat, ctx.qp, ctx.bypass, is_b, nref, depth,
            );
            let j_n = ssd_n as f64 + lam * f64::from(bits_n);
            if j_n < j_i.min(j) {
                self.recycle(coded);
                self.saves.extend([mid, saved]);
                return (nxn, ssd_n, bits_n);
            }
            self.restore(mid, x0, y0, n);
        }
        if j_i < j {
            self.recycle(coded);
            self.saves.push(saved);
            return (intra, ssd_i, bits_i);
        }
        self.restore(saved, x0, y0, n);
        (coded, ssd, bits)
    }

    /// The unit [`Self::tree_leaf_coded`] decided, or the same unit without
    /// its residual where that is cheaper: a merge unit becomes a skip, an
    /// AMVP unit codes `rqt_root_cbf` 0. Coefficients the quantiser kept
    /// are not always worth their bits, and only the comparison in
    /// distortion plus rate can say which.
    #[allow(clippy::too_many_arguments)]
    fn tree_leaf_drop(
        &mut self,
        ctx: &MeCtx<'_, S>,
        refs: TreeRefs<'_, S>,
        x0: usize,
        y0: usize,
        log2: u32,
        depth: u32,
        src: &Srcs<'_, S>,
        coded: PCuDecision,
        ssd: u64,
        bits: f32,
    ) -> (PCuDecision, u64, f32) {
        let PCuDecision::Inter(d) = &coded else {
            return (coded, ssd, bits);
        };
        if !d.rqt_root_cbf || ctx.bypass || d.part != PartMode::P2Nx2N {
            return (coded, ssd, bits);
        }
        let pick = match d.kind {
            InterCuKind::Merge { merge_idx } => PuPick {
                merge_idx: Some(merge_idx),
                mv: [d.mv, d.mv_l1],
                ref_idx: [d.ref_idx, d.ref_idx_l1],
                ..PuPick::default()
            },
            InterCuKind::Amvp { mvp_flag, mvd } => PuPick {
                mvd: [mvd, Mv::ZERO],
                mvp_flag: [mvp_flag, 0],
                mv: [d.mv, d.mv_l1],
                ref_idx: [d.ref_idx, d.ref_idx_l1],
                ..PuPick::default()
            },
            InterCuKind::BAmvp { idc, mvd, mvp_flag } => PuPick {
                idc,
                mvd,
                mvp_flag,
                mv: [d.mv, d.mv_l1],
                ref_idx: [d.ref_idx, d.ref_idx_l1],
                ..PuPick::default()
            },
            _ => return (coded, ssd, bits),
        };
        let (nref, is_b) = match refs {
            TreeRefs::P(l0) => (l0.len() as u32, false),
            TreeRefs::B(..) => (1, true),
        };
        let n = 1usize << log2;
        let saved = self.take_save(x0, y0, n);
        let mut alt = self.fresh(log2, ctx);
        self.no_residual = true;
        self.code_whole(
            ctx,
            refs,
            &mut alt,
            pick,
            x0,
            y0,
            log2,
            src.y,
            src.y_stride,
            src.cb,
            src.cr,
            src.c_stride,
        );
        self.no_residual = false;
        let alt = PCuDecision::Inter(alt);
        if !matches!(&alt, PCuDecision::Inter(a) if a.kind == InterCuKind::UseIntra) {
            let ssd_a = cu_ssd(ctx, &self.recon, self.cat, x0, y0, n, src);
            let bits_a = crate::encode::h265::p_cu_bits(
                &alt, self.cat, ctx.qp, ctx.bypass, is_b, nref, depth,
            );
            let lam = ssd_lambda(ctx.qp, ctx.bit_depth);
            if (ssd_a as f64) + lam * f64::from(bits_a) < (ssd as f64) + lam * f64::from(bits) {
                self.recycle(coded);
                self.saves.push(saved);
                return (alt, ssd_a, bits_a);
            }
        }
        self.recycle(alt);
        self.restore(saved, x0, y0, n);
        (coded, ssd, bits)
    }

    #[allow(clippy::too_many_arguments)]
    fn tree_leaf_coded(
        &mut self,
        ctx: &MeCtx<'_, S>,
        refs: TreeRefs<'_, S>,
        x0: usize,
        y0: usize,
        log2: u32,
        depth: u32,
        src: &Srcs<'_, S>,
    ) -> (PCuDecision, u64, f32) {
        let mut d = self.fresh(log2, ctx);
        self.tree_bits = None;
        let (nref, is_b) = match refs {
            TreeRefs::P(l0) => {
                self.code_cu_into(
                    ctx,
                    l0,
                    x0,
                    y0,
                    log2,
                    depth,
                    src.y,
                    src.y_stride,
                    src.cb,
                    src.cr,
                    src.c_stride,
                    &mut d,
                );
                (l0.len() as u32, false)
            }
            TreeRefs::B(r0, r1) => {
                self.code_cu_b_into(
                    ctx,
                    r0,
                    r1,
                    x0,
                    y0,
                    log2,
                    depth,
                    src.y,
                    src.y_stride,
                    src.cb,
                    src.cr,
                    src.c_stride,
                    &mut d,
                );
                (1, true)
            }
        };
        if self.gated {
            return (PCuDecision::Inter(d), 0, f32::INFINITY);
        }
        let coded = if matches!(d.kind, InterCuKind::UseIntra) {
            self.pool.push(d);
            PCuDecision::Intra(Box::new(self.code_cu_intra(
                ctx,
                x0,
                y0,
                log2,
                src.y,
                src.y_stride,
                src.cb,
                src.cr,
                src.c_stride,
            )))
        } else {
            PCuDecision::Inter(d)
        };
        let n = 1usize << log2;
        let ssd = cu_ssd(ctx, &self.recon, self.cat, x0, y0, n, src);
        let bits = match (&coded, self.tree_bits.take()) {
            (PCuDecision::Inter(d), Some(tree)) => crate::encode::h265::p_cu_bits_with_tree(
                d, ctx.qp, ctx.bypass, is_b, nref, depth, tree,
            ),
            _ => crate::encode::h265::p_cu_bits(
                &coded, self.cat, ctx.qp, ctx.bypass, is_b, nref, depth,
            ),
        };
        // A 2Nx2N unit and a partitioned shape were chosen between in SATD
        // plus bits (`code_cu`); when the loser came close, code it too and
        // keep the cheaper in SSD plus lambda times the CU's bits, the
        // loser's state put back from a copy. See `PARTS_RD_MARGIN`.
        if let (Some((whole, Some((part, pus, parts_cost)))), PCuDecision::Inter(d)) =
            (self.alts.take(), &coded)
        {
            let chose_parts = d.kind == InterCuKind::Parts;
            let (mine, other) = if chose_parts {
                (parts_cost, whole.cost)
            } else {
                (whole.cost, parts_cost)
            };
            if other <= mine * (1.0 + PARTS_RD_MARGIN) {
                let saved = self.take_save(x0, y0, n);
                let mut alt = self.fresh(log2, ctx);
                if chose_parts {
                    self.code_whole(
                        ctx,
                        refs,
                        &mut alt,
                        whole,
                        x0,
                        y0,
                        log2,
                        src.y,
                        src.y_stride,
                        src.cb,
                        src.cr,
                        src.c_stride,
                    )
                } else {
                    self.code_parts(
                        ctx,
                        refs,
                        &mut alt,
                        part,
                        pus,
                        x0,
                        y0,
                        log2,
                        src.y,
                        src.y_stride,
                        src.cb,
                        src.cr,
                        src.c_stride,
                    );
                }
                let alt = PCuDecision::Inter(alt);
                if !matches!(&alt, PCuDecision::Inter(a) if a.kind == InterCuKind::UseIntra) {
                    let ssd_a = cu_ssd(ctx, &self.recon, self.cat, x0, y0, n, src);
                    let bits_a = crate::encode::h265::p_cu_bits(
                        &alt, self.cat, ctx.qp, ctx.bypass, is_b, nref, depth,
                    );
                    let lam = ssd_lambda(ctx.qp, ctx.bit_depth);
                    if (ssd_a as f64) + lam * f64::from(bits_a)
                        < (ssd as f64) + lam * f64::from(bits)
                    {
                        self.recycle(coded);
                        self.saves.push(saved);
                        return (alt, ssd_a, bits_a);
                    }
                }
                self.recycle(alt);
                self.restore(saved, x0, y0, n);
            }
        }
        // An intra unit at the minimum coding block has a second shape, as
        // in an I picture: code `PART_NxN` too and keep the cheaper, the
        // loser's state put back from a copy.
        if log2 == MIN_CB_LOG2 && matches!(coded, PCuDecision::Intra(_)) && self.nxn_trial > 0.0 {
            let saved = self.take_save(x0, y0, n);
            let nxn = PCuDecision::Intra(Box::new(self.code_cu_intra_nxn(
                ctx,
                x0,
                y0,
                src.y,
                src.y_stride,
                src.cb,
                src.cr,
                src.c_stride,
            )));
            let ssd_n = cu_ssd(ctx, &self.recon, self.cat, x0, y0, n, src);
            let bits_n = crate::encode::h265::p_cu_bits(
                &nxn, self.cat, ctx.qp, ctx.bypass, is_b, nref, depth,
            );
            let lam = ssd_lambda(ctx.qp, ctx.bit_depth);
            if ssd_n as f64 + lam * f64::from(bits_n) < ssd as f64 + lam * f64::from(bits) {
                self.recycle(coded);
                self.saves.push(saved);
                return (nxn, ssd_n, bits_n);
            }
            self.restore(saved, x0, y0, n);
        }
        (coded, ssd, bits)
    }

    /// The `PART_NxN` alternative to [`Self::code_cu_intra`] for an 8x8 unit
    /// the inter decision handed over: `code_cu_nxn_intra` over this
    /// picture's state, as `code_cu_intra` runs `code_cu_2nx2n_intra`.
    #[allow(clippy::too_many_arguments)]
    pub fn code_cu_intra_nxn(
        &mut self,
        ctx: &MeCtx<'_, S>,
        x0: usize,
        y0: usize,
        src_y: &[S],
        y_stride: usize,
        src_cb: &[S],
        src_cr: &[S],
        c_stride: usize,
    ) -> CuDecision {
        let PicInfo {
            intra_mode,
            pred_mode,
            ..
        } = &mut self.info;
        let (intra_mode, pred_mode) = (&mut intra_mode[..], &pred_mode[..]);
        code_cu_nxn_intra(
            ctx,
            self.geo,
            &mut self.recon,
            intra_mode,
            Some(pred_mode),
            &mut self.intra_scratch,
            x0,
            y0,
            src_y,
            y_stride,
            src_cb,
            src_cr,
            c_stride,
        )
    }

    /// The half-sample planes of the reference at `poc`, if the caller
    /// supplied them.
    fn half_for(&self, poc: i32) -> Option<std::sync::Arc<h265_search::HalfPlanes<S>>> {
        self.half
            .iter()
            .find(|(p, _)| *p == poc)
            .map(|(_, h)| h.clone())
    }

    /// The samples a uni-predicted block at `(x, y)` of `wh` under default
    /// weighting reads at `mv` from `refp`, when they can be read rather
    /// than interpolated: the reference itself at a whole-sample vector,
    /// its precomputed fractional phase otherwise (`self.half`, found by
    /// the plane's address). `None` sends the caller to the
    /// interpolating path.
    fn read_at<'a>(
        half: &'a [(i32, std::sync::Arc<h265_search::HalfPlanes<S>>)],
        refp: &'a Plane16<S>,
        x: usize,
        y: usize,
        (w, h): (usize, usize),
        mv: Mv,
    ) -> Option<(&'a [S], usize)> {
        let (mvx, mvy) = (i32::from(mv.x), i32::from(mv.y));
        if mvx & 3 == 0 && mvy & 3 == 0 {
            let (xi, yi) = (x as i32 + (mvx >> 2), y as i32 + (mvy >> 2));
            let lim = refp.pad as i32;
            if xi < -lim
                || yi < -lim
                || xi + w as i32 > refp.width as i32 + lim
                || yi + h as i32 > refp.height as i32 + lim
            {
                return None;
            }
            let off = refp.offset(xi as isize, yi as isize);
            return Some((&refp.data[off..], refp.stride));
        }
        let key = refp.data.as_ptr() as usize;
        let hp = half.iter().find(|(_, h)| h.source == key)?;
        hp.1.at(x, y, w, h, mvx, mvy)
    }

    /// Whether reference `r` is searched for `pu`: always the nearest, and
    /// under [`Self::limit_refs`] below the top of the tree only the one
    /// the enclosing unit chose — a unit's best reference is most often
    /// its parent's.
    fn ref_searched(&self, pu: &PuPos, r: usize) -> bool {
        if r == 0 || !self.limit_refs {
            return true;
        }
        let d = self.depth_of(pu);
        if d == 0 {
            return true;
        }
        match self.parent_ref[d - 1] {
            Some((px, py, pr)) => {
                let n = (pu.n_cb as u32) * 2;
                let inside =
                    (pu.x_cb as u32).wrapping_sub(px) < n && (pu.y_cb as u32).wrapping_sub(py) < n;
                !inside || pr as usize == r
            }
            None => true,
        }
    }

    /// The coding-tree depth of the unit `pu` belongs to.
    fn depth_of(&self, pu: &PuPos) -> usize {
        (self.log2_ctb - (pu.n_cb as u32).trailing_zeros()) as usize
    }

    /// The vector the enclosing unit one depth up found for (`list`,
    /// `r`), when it was searched over this unit's position.
    fn parent_seed(&self, pu: &PuPos, list: usize, r: usize) -> Option<Mv> {
        let d = self.depth_of(pu);
        if d == 0 || r >= 8 {
            return None;
        }
        let (px, py, mv) = self.seed_at[d - 1][list * 8 + r]?;
        let n = (pu.n_cb as u32) * 2;
        let inside = (pu.x_cb as u32).wrapping_sub(px) < n && (pu.y_cb as u32).wrapping_sub(py) < n;
        inside.then_some(mv)
    }

    /// Record what this unit's search found, for the units below it.
    fn note_seed(&mut self, pu: &PuPos, list: usize, r: usize, mv: Mv) {
        let d = self.depth_of(pu);
        if d < 4 && r < 8 && pu.w == pu.n_cb && pu.h == pu.n_cb {
            self.seed_at[d][list * 8 + r] = Some((pu.x_cb as u32, pu.y_cb as u32, mv));
        }
    }

    /// Full-sample search: every seed (rounded toward zero to full
    /// samples, as the decoder's `>> 2` addresses them) scored, then a
    /// small-diamond descent from the best, widened by square rings at 4,
    /// 8 and 16 samples whenever the descent settles, so a unit whose
    /// motion no seed predicted can still find it. Every candidate is
    /// priced SAD plus `lam` times the bits of its difference from the
    /// cheaper of `mvp` (`h265_search::mv_bits`): without the rate term
    /// the search returns the vector of least distortion however much it
    /// costs to say. Returns the best vector in quarter units.
    #[allow(clippy::too_many_arguments)]
    fn search_full(
        &self,
        ctx: &MeCtx<'_, S>,
        refp: &Plane16<S>,
        x: usize,
        y: usize,
        (w, h): (usize, usize),
        src: &[S],
        src_stride: usize,
        seeds: &[Mv],
        lam: f32,
        mvp: [Mv; 2],
    ) -> Mv {
        let clamp_pos = |fx: i32, fy: i32| -> (i32, i32) {
            let pad = (refp.pad - h265_search::MARGIN) as i32;
            let xi = (x as i32 + fx).clamp(-pad, refp.width as i32 + pad - w as i32);
            let yi = (y as i32 + fy).clamp(-pad, refp.height as i32 + pad - h as i32);
            (xi - x as i32, yi - y as i32)
        };
        let pq = [
            (i32::from(mvp[0].x), i32::from(mvp[0].y)),
            (i32::from(mvp[1].x), i32::from(mvp[1].y)),
        ];
        let cost_of = |fx: i32, fy: i32| -> u32 {
            let off = refp.offset((x as i32 + fx) as isize, (y as i32 + fy) as isize);
            let sad = (ctx.dist.sad)(src, src_stride, &refp.data[off..], refp.stride, w, h);
            sad + (lam * h265_search::mv_bits(fx * 4, fy * 4, &pq) as f32) as u32
        };
        let mut best = (0i32, 0i32);
        let mut best_cost = u32::MAX;
        let mut tried: [(i32, i32); 16] = [(i32::MIN, 0); 16];
        for (i, s) in seeds.iter().enumerate() {
            let c = clamp_pos(s.x as i32 >> 2, s.y as i32 >> 2);
            if tried.contains(&c) {
                continue;
            }
            if i < tried.len() {
                tried[i] = c;
            }
            let cost = cost_of(c.0, c.1);
            if cost < best_cost {
                best_cost = cost;
                best = c;
            }
        }
        let centre = best;
        let inside = |c: (i32, i32)| {
            (c.0 - centre.0).abs() <= SEARCH_RANGE
                && (c.1 - centre.1).abs() <= SEARCH_RANGE
                && clamp_pos(c.0, c.1) == c
        };
        let mut rings = 0;
        loop {
            // Small diamond to convergence.
            for _ in 0..(2 * SEARCH_RANGE) {
                let mut improved = false;
                let from = best;
                for (dx, dy) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                    let cand = (from.0 + dx, from.1 + dy);
                    if !inside(cand) {
                        continue;
                    }
                    let cost = cost_of(cand.0, cand.1);
                    if cost < best_cost {
                        best_cost = cost;
                        best = cand;
                        improved = true;
                    }
                }
                if !improved {
                    break;
                }
            }
            if rings >= self.search_rings {
                break;
            }
            rings += 1;
            // Square rings around the settled point; a better point
            // restarts the descent from there.
            let from = best;
            let mut moved = false;
            for r in [4i32, 8, 16] {
                for (dx, dy) in [
                    (r, 0),
                    (-r, 0),
                    (0, r),
                    (0, -r),
                    (r, r),
                    (-r, r),
                    (r, -r),
                    (-r, -r),
                ] {
                    let cand = (from.0 + dx, from.1 + dy);
                    if !inside(cand) {
                        continue;
                    }
                    let cost = cost_of(cand.0, cand.1);
                    if cost < best_cost {
                        best_cost = cost;
                        best = cand;
                        moved = true;
                    }
                }
            }
            if !moved {
                break;
            }
        }
        Mv::new((best.0 * 4) as i16, (best.1 * 4) as i16)
    }

    /// Sub-sample refinement around `start`: the eight half-sample
    /// neighbours, then the eight quarter-sample neighbours of the winner,
    /// each scored SATD plus `lam` times its vector difference's bits from
    /// the cheaper of `mvp`. Half-sample candidates read `half` — the
    /// reference's exact half-sample phases — when it is given and the
    /// unit predicts under default weighting; everything else
    /// interpolates. Returns the vector and its SATD (without the rate).
    #[allow(clippy::too_many_arguments)]
    fn refine_subpel(
        &mut self,
        ctx: &MeCtx<'_, S>,
        refp: &Plane16<S>,
        half: Option<&h265_search::HalfPlanes<S>>,
        wp: Weighting,
        list: usize,
        x: usize,
        y: usize,
        wh: (usize, usize),
        src: &[S],
        src_stride: usize,
        start: Mv,
        lam: f32,
        mvp: [Mv; 2],
    ) -> (Mv, u32) {
        let pq = [
            (i32::from(mvp[0].x), i32::from(mvp[0].y)),
            (i32::from(mvp[1].x), i32::from(mvp[1].y)),
        ];
        let rate = |mv: Mv| {
            (lam * h265_search::mv_bits(i32::from(mv.x), i32::from(mv.y), &pq) as f32) as u32
        };
        let half = half.filter(|_| matches!(wp, Weighting::Default));
        let mut best = start;
        let mut best_satd =
            self.satd_at_weighted(ctx, refp, x, y, wh, src, src_stride, start, wp, list);
        let mut best_cost = best_satd + rate(start);
        for step in [2i16, 1] {
            let centre = best;
            let cross = step == 1 && self.quarter_cross;
            for dy in [-step, 0, step] {
                for dx in [-step, 0, step] {
                    if (dx == 0 && dy == 0) || (cross && dx != 0 && dy != 0) {
                        continue;
                    }
                    let mv = Mv::new(centre.x.wrapping_add(dx), centre.y.wrapping_add(dy));
                    let satd = match half
                        .and_then(|hp| hp.at(x, y, wh.0, wh.1, i32::from(mv.x), i32::from(mv.y)))
                    {
                        Some((p, ps)) => (ctx.dist.satd)(src, src_stride, p, ps, wh.0, wh.1),
                        None => self
                            .satd_at_weighted(ctx, refp, x, y, wh, src, src_stride, mv, wp, list),
                    };
                    let cost = satd + rate(mv);
                    if cost < best_cost {
                        best_cost = cost;
                        best_satd = satd;
                        best = mv;
                    }
                }
            }
        }
        (best, best_satd)
    }

    /// One round of iterative bi-prediction refinement for list `list` of
    /// a B unit predicting from both lists at `mvs`: the other list's
    /// prediction held fixed, list `list` is searched again — whole
    /// samples from its current vector, then the sub-sample rings —
    /// against the target `2 * src - other`, whose difference from a
    /// candidate's prediction is twice the bi-predicted residual (so the
    /// vector rate is weighed at twice `lam`). Default weighting only.
    /// Returns the vector found; the caller scores the pair exactly.
    #[allow(clippy::too_many_arguments)]
    fn refine_bi(
        &mut self,
        ctx: &MeCtx<'_, S>,
        planes: [&Plane16<S>; 2],
        half: Option<&h265_search::HalfPlanes<S>>,
        list: usize,
        x: usize,
        y: usize,
        (w, h): (usize, usize),
        src: &[S],
        src_stride: usize,
        mvs: [Mv; 2],
        lam: f32,
        mvp: [Mv; 2],
    ) -> Mv {
        let other = 1 - list;
        let mut t = std::mem::take(&mut self.bi_target);
        let max = (1i32 << ctx.bit_depth) - 1;
        {
            let (p, ps) = match Self::read_at(&self.half, planes[other], x, y, (w, h), mvs[other]) {
                Some(v) => v,
                None => {
                    let InterPicture {
                        swin,
                        stmp,
                        spred14,
                        spred,
                        ..
                    } = self;
                    predict14(
                        ctx,
                        planes[other],
                        x,
                        y,
                        (w, h),
                        mvs[other],
                        swin,
                        stmp,
                        spred14,
                    );
                    let bd = ctx.bit_depth;
                    (ctx.dsp.uni)(spred, w, spred14, w, h, 14 - bd as i32, max);
                    (&self.spred[..], w)
                }
            };
            for yy in 0..h {
                for xx in 0..w {
                    let v = 2 * src[yy * src_stride + xx].to_i32() - p[yy * ps + xx].to_i32();
                    t[yy * w + xx] = S::from_i32(v.clamp(0, max));
                }
            }
        }
        let full = self.search_full(
            ctx,
            planes[list],
            x,
            y,
            (w, h),
            &t,
            w,
            &[mvs[list]],
            2.0 * lam,
            mvp,
        );
        let (mv, _) = self.refine_subpel(
            ctx,
            planes[list],
            half,
            Weighting::Default,
            list,
            x,
            y,
            (w, h),
            &t,
            w,
            full,
            2.0 * lam,
            mvp,
        );
        self.bi_target = t;
        mv
    }

    #[allow(clippy::too_many_arguments)]
    fn satd_at(
        &mut self,
        ctx: &MeCtx<'_, S>,
        refp: &Plane16<S>,
        r: usize,
        x: usize,
        y: usize,
        wh: (usize, usize),
        src: &[S],
        src_stride: usize,
        mv: Mv,
    ) -> u32 {
        let wp = self.wp_for(r)[0];
        self.satd_at_weighted(ctx, refp, x, y, wh, src, src_stride, mv, wp, 0)
    }

    /// [`Self::satd_at`] under a given luma weighting: the default
    /// uni-prediction, or the decoder's `weighted_uni` at the table's
    /// `log2WD` and the weight and offset of `list` — the entry
    /// `predict_block` takes for a one-list prediction from that list, and
    /// the same kernel it will commit the winner through.
    #[allow(clippy::too_many_arguments)]
    fn satd_at_weighted(
        &mut self,
        ctx: &MeCtx<'_, S>,
        refp: &Plane16<S>,
        x: usize,
        y: usize,
        (w, h): (usize, usize),
        src: &[S],
        src_stride: usize,
        mv: Mv,
        wp: Weighting,
        list: usize,
    ) -> u32 {
        if matches!(wp, Weighting::Default)
            && let Some((p, ps)) = Self::read_at(&self.half, refp, x, y, (w, h), mv)
        {
            return (ctx.dist.satd)(src, src_stride, p, ps, w, h);
        }
        let InterPicture {
            swin,
            stmp,
            spred14,
            spred,
            ..
        } = self;
        predict14(ctx, refp, x, y, (w, h), mv, swin, stmp, spred14);
        let bd = ctx.bit_depth;
        let max = (1i32 << bd) - 1;
        match wp {
            Weighting::Default => (ctx.dsp.uni)(spred, w, spred14, w, h, 14 - bd as i32, max),
            Weighting::Explicit { log2_wd, w: wt, o } => {
                (ctx.dsp.weighted_uni)(spred, w, spred14, w, h, log2_wd, wt[list], o[list], max)
            }
        }
        (ctx.dist.satd)(src, src_stride, spred, w, w, h)
    }

    /// The model check for weighted prediction: the luma SATD of the CU
    /// at `(x0, y0)` predicted from `refp` at `mv` without any weighting
    /// and with this picture's, so the caller can count whether the
    /// table's fit helped the vectors the search actually chose.
    #[allow(clippy::too_many_arguments)]
    pub fn weighting_gain(
        &mut self,
        ctx: &MeCtx<'_, S>,
        refp: &Frame<S>,
        r: usize,
        x0: usize,
        y0: usize,
        log2_cu: u32,
        src_y: &[S],
        y_stride: usize,
        mv: Mv,
    ) -> (u32, u32) {
        let n = 1usize << log2_cu;
        let src = &src_y[y0 * y_stride + x0..];
        let wp = self.wp_for(r)[0];
        let plain = self.satd_at_weighted(
            ctx,
            &refp.y,
            x0,
            y0,
            (n, n),
            src,
            y_stride,
            mv,
            Weighting::Default,
            0,
        );
        let weighted =
            self.satd_at_weighted(ctx, &refp.y, x0, y0, (n, n), src, y_stride, mv, wp, 0);
        (plain, weighted)
    }

    /// [`Self::weighting_gain`] for a B CU: the luma SATD at the chosen
    /// vectors `mv` of the lists `ref_idx` uses (reference 0 of each, `ref0`
    /// and `ref1`), without weighting and with this picture's — one list's
    /// prediction, or the pair's.
    #[allow(clippy::too_many_arguments)]
    pub fn weighting_gain_b(
        &mut self,
        ctx: &MeCtx<'_, S>,
        ref0: &Frame<S>,
        ref1: &Frame<S>,
        x0: usize,
        y0: usize,
        log2_cu: u32,
        src_y: &[S],
        y_stride: usize,
        mv: [Mv; 2],
        ref_idx: [i8; 2],
    ) -> (u32, u32) {
        let n = 1usize << log2_cu;
        let src = &src_y[y0 * y_stride + x0..];
        let wp = self.wp_b_for(ref_idx)[0];
        match (ref_idx[0] >= 0, ref_idx[1] >= 0) {
            (true, true) => {
                let plain = self.satd_bi_at(
                    ctx,
                    &ref0.y,
                    &ref1.y,
                    x0,
                    y0,
                    (n, n),
                    src,
                    y_stride,
                    mv[0],
                    mv[1],
                    Weighting::Default,
                );
                (
                    plain,
                    self.satd_bi_at(
                        ctx,
                        &ref0.y,
                        &ref1.y,
                        x0,
                        y0,
                        (n, n),
                        src,
                        y_stride,
                        mv[0],
                        mv[1],
                        wp,
                    ),
                )
            }
            (one0, _) => {
                let (refp, list) = if one0 { (&ref0.y, 0) } else { (&ref1.y, 1) };
                let plain = self.satd_at_weighted(
                    ctx,
                    refp,
                    x0,
                    y0,
                    (n, n),
                    src,
                    y_stride,
                    mv[list],
                    Weighting::Default,
                    list,
                );
                (
                    plain,
                    self.satd_at_weighted(
                        ctx,
                        refp,
                        x0,
                        y0,
                        (n, n),
                        src,
                        y_stride,
                        mv[list],
                        wp,
                        list,
                    ),
                )
            }
        }
    }

    /// SATD of the *bi-predicted* luma block at `(mv0, mv1)`: the two
    /// lists' 14-bit predictions combined through the decoder's own
    /// `dsp.bi` at `15 - bit_depth`, which is the shift `predict_block`
    /// uses for default-weighted bi-prediction (8.5.3.3.4.2), or under
    /// `wp` through its `dsp.weighted_bi` at the table's `log2WD` and both
    /// lists' weights and offsets (8.5.3.3.4.3) — the arithmetic
    /// `predict_block` commits the winner through. Scoring the average
    /// rather than either half is what makes the BI trial comparable with
    /// the two uni ones.
    #[allow(clippy::too_many_arguments)]
    fn satd_bi_at(
        &mut self,
        ctx: &MeCtx<'_, S>,
        ref0: &Plane16<S>,
        ref1: &Plane16<S>,
        x: usize,
        y: usize,
        (w, h): (usize, usize),
        src: &[S],
        src_stride: usize,
        mv0: Mv,
        mv1: Mv,
        wp: Weighting,
    ) -> u32 {
        if matches!(wp, Weighting::Default)
            && let (Some((p0, s0)), Some((p1, s1))) = (
                Self::read_at(&self.half, ref0, x, y, (w, h), mv0),
                Self::read_at(&self.half, ref1, x, y, (w, h), mv1),
            )
        {
            // The two one-list predictions averaged at sample precision:
            // within one of the bi-prediction 8.5.3.3.4.2 makes from the
            // 14-bit intermediates, which is close enough to rank
            // candidates by; the unit that wins is predicted exactly.
            let avg = &mut self.spred_avg;
            for yy in 0..h {
                for xx in 0..w {
                    avg[yy * w + xx] = S::from_i32(
                        (p0[yy * s0 + xx].to_i32() + p1[yy * s1 + xx].to_i32() + 1) >> 1,
                    );
                }
            }
            return (ctx.dist.satd)(src, src_stride, &self.spred_avg, w, w, h);
        }
        let InterPicture {
            swin,
            stmp,
            spred14,
            spred14_b,
            spred,
            ..
        } = self;
        predict14(ctx, ref0, x, y, (w, h), mv0, swin, stmp, spred14);
        predict14(ctx, ref1, x, y, (w, h), mv1, swin, stmp, spred14_b);
        let bd = ctx.bit_depth;
        let max = (1i32 << bd) - 1;
        match wp {
            Weighting::Default => {
                (ctx.dsp.bi)(spred, w, spred14, spred14_b, w, h, 15 - bd as i32, max)
            }
            Weighting::Explicit { log2_wd, w: wt, o } => (ctx.dsp.weighted_bi)(
                spred, w, spred14, spred14_b, w, h, log2_wd, wt[0], wt[1], o[0], o[1], max,
            ),
        }
        (ctx.dist.satd)(src, src_stride, spred, w, w, h)
    }
}

/// `refp` with an explicit luma weighting applied to every sample, border
/// included: 8.5.3.3.4.3's one-list formula at a whole-sample vector — the
/// sample at 14 bits (`<< (14 - bitDepth)`), then `((p * w + 2^(log2WD -
/// 1)) >> log2WD) + o`, clipped — with `log2WD`, `w` and `o` as
/// `explicit_weighting` derives them. The plane a weighted list's
/// full-sample search scores against ([`InterPicture::search_b`]): a search
/// only ranks candidates, and at whole samples this is exactly the
/// prediction `predict_block` would make.
fn weighted_search_plane<S: Sample>(
    refp: &Plane16<S>,
    bit_depth: u32,
    log2_wd: i32,
    w: i32,
    o: i32,
) -> Plane16<S> {
    let mut out = refp.clone();
    let (shift, max, round) = (
        14 - bit_depth as i32,
        (1i32 << bit_depth) - 1,
        1i32 << (log2_wd - 1),
    );
    for v in out.data.iter_mut() {
        let p = (((v.to_i32() << shift) * w + round) >> log2_wd) + o;
        *v = S::from_i32(p.clamp(0, max));
    }
    out
}

/// The references a coding-quadtree walk predicts from.
pub enum TreeRefs<'r, S: Sample> {
    /// A P slice's `RefPicList0`, nearest first.
    P(&'r [&'r Frame<S>]),
    /// A B slice's list-0 and list-1 anchors.
    B(&'r Frame<S>, &'r Frame<S>),
}

impl<S: Sample> Clone for TreeRefs<'_, S> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<S: Sample> Copy for TreeRefs<'_, S> {}

/// Everything a coding trial over a square region of an inter picture
/// writes, as the trial left it: the samples, the motion grid, and the
/// `pred_mode`, `skip` and `intra_mode` marks. See
/// [`InterPicture::code_ctu_tree`].
struct TrialSave<S: Sample> {
    planes: RegionSave<S>,
    motion: Vec<MotionInfo>,
    pred_mode: Vec<u8>,
    skip: Vec<u8>,
    intra_mode: Vec<u8>,
}

impl<S: Sample> TrialSave<S> {
    fn empty() -> Self {
        TrialSave {
            planes: RegionSave::empty(),
            motion: Vec::new(),
            pred_mode: Vec::new(),
            skip: Vec::new(),
            intra_mode: Vec::new(),
        }
    }

    fn fill(&mut self, pic: &InterPicture<S>, x0: usize, y0: usize, n: usize) {
        let w4 = pic.info.w4;
        self.planes.take_into(&pic.recon, pic.cat, x0, y0, n);
        save4_into(&pic.recon.motion, pic.recon.w4, x0, y0, n, &mut self.motion);
        save4_into(&pic.info.pred_mode, w4, x0, y0, n, &mut self.pred_mode);
        save4_into(&pic.info.skip, w4, x0, y0, n, &mut self.skip);
        save4_into(&pic.info.intra_mode, w4, x0, y0, n, &mut self.intra_mode);
    }

    fn put(&self, pic: &mut InterPicture<S>, x0: usize, y0: usize, n: usize) {
        let w4 = pic.info.w4;
        self.planes.put(&mut pic.recon, pic.cat, x0, y0, n);
        let mw4 = pic.recon.w4;
        restore4(&mut pic.recon.motion, mw4, x0, y0, n, &self.motion);
        restore4(&mut pic.info.pred_mode, w4, x0, y0, n, &self.pred_mode);
        restore4(&mut pic.info.skip, w4, x0, y0, n, &self.skip);
        restore4(&mut pic.info.intra_mode, w4, x0, y0, n, &self.intra_mode);
    }
}

/// One list's 14-bit luma prediction of the `w x h` block at `(x, y)` for
/// vector `mv`, into `out` (stride `w`). The window gathering and the kernel choice are
/// `hevc::inter`'s `source` and `interp` (3 taps before the sample, 4
/// after; `qpel_copy` / `_h` / `_v` / `_h`+`_v2` by fraction), which is
/// what keeps a scored candidate identical to what `predict_block` will
/// later commit for the winner.
///
/// A free function rather than a method because both callers need it to
/// write into one field of `InterPicture` while reading two others, which
/// a `&mut self` method cannot express.
#[allow(clippy::too_many_arguments)]
fn predict14<S: Sample>(
    ctx: &MeCtx<'_, S>,
    refp: &Plane16<S>,
    x: usize,
    y: usize,
    (w, h): (usize, usize),
    mv: Mv,
    swin: &mut [S],
    stmp: &mut [i16],
    out: &mut [i16],
) {
    let xi = x as i32 + (mv.x as i32 >> 2);
    let yi = y as i32 + (mv.y as i32 >> 2);
    let (fx, fy) = ((mv.x & 3) as usize, (mv.y & 3) as usize);
    let bd = ctx.bit_depth;
    let shift1 = bd.min(12) as i32 - 8;
    let shift3 = 14 - bd as i32;
    // The filter window (3 samples before, 4 after), gathered with
    // clamping when it leaves the padded plane — `source`'s rule.
    let (x0, y0) = (xi - 3, yi - 3);
    let (ww, hh) = (w + 7, h + 7);
    let pad = refp.pad as i32;
    let inside = x0 >= -pad
        && y0 >= -pad
        && x0 + ww as i32 <= refp.width as i32 + pad
        && y0 + hh as i32 <= refp.height as i32 + pad;
    let (win, stride) = if inside {
        (
            &refp.data[refp.offset(x0 as isize, y0 as isize)..],
            refp.stride,
        )
    } else {
        for yy in 0..hh {
            for xx in 0..ww {
                swin[yy * ww + xx] = refp.at_clamped(x0 + xx as i32, y0 + yy as i32);
            }
        }
        (&swin[..], ww)
    };
    let at_block = 3 * stride + 3;
    match (fx, fy) {
        (0, 0) => (ctx.dsp.qpel_copy)(out, &win[at_block..], stride, w, h, shift3),
        (_, 0) => (ctx.dsp.qpel_h)(out, &win[3 * stride..], stride, w, h, fx, shift1),
        (0, _) => (ctx.dsp.qpel_v)(out, &win[3..], stride, w, h, fy, shift1),
        _ => {
            (ctx.dsp.qpel_h)(stmp, win, stride, w, h + 7, fx, shift1);
            (ctx.dsp.qpel_v2)(out, stmp, w, w, h, fy);
        }
    }
}

/// The Lagrangian for a cost whose distortion is a SATD (or SAD): the
/// square root of the SSD Lagrangian `0.85 * 2^((QP − 12) / 3)`
/// ([`super::h265_intra::satd_lambda`]). Callers multiply by
/// `satd_lambda_scale` for deeper samples.
fn lambda(qp: i32) -> f32 {
    super::h265_intra::satd_lambda(qp)
}

/// What a candidate shape costs to *signal*, in real bits.
///
/// This replaces the hand-rolled bin counts this module used to carry —
/// `tr_bins`, and an `mvd_cost` that approximated `write_mvd`'s
/// exponential-Golomb remainder as `5 + 2 * log2(a - 1)`. The trouble with
/// those was never accuracy: it was that nothing could check them. A wrong
/// cost changes which shape wins, and every check this project has — SELF,
/// CROSS, the replays — passes whatever the decision picks.
///
/// So the bins are no longer guessed at. Each shape is priced by running
/// **the production writers** through a counting [`CabacEncoder`], which
/// tallies exactly the bits it would have written (an equality asserted in
/// `cabac_enc`'s round trip). `write_mvd`'s Golomb coding, `write_merge_idx`'s
/// truncated unary and its `MaxNumMergeCand` cap, the context each bin
/// lands in — all of it is the real thing rather than a model of it.
///
/// # What this is not, stated precisely
///
/// **The probabilities are the slice's initial ones, not the ones in force
/// at this CU.** The decision runs in its own pass, before serialisation —
/// SAO forced that split — so the live context array does not exist yet
/// when a shape is chosen. Pricing therefore starts from a freshly
/// initialised `Contexts` for this slice type and QP. The bin *sequence*
/// is exact; the bit *width* of each bin is priced under the slice's
/// starting model rather than its adapted one.
///
/// **The neighbour-dependent contexts are the neutral ones.**
/// `cu_skip_flag`'s context counts skipped neighbours and
/// `split_cu_flag`'s counts deeper ones; both are serialiser state. They
/// are priced here as if no neighbour were available, which is what the
/// first CU of a slice genuinely sees.
///
/// **Residual bits are not included**, matching the scope of the counts it
/// replaces: at shape-choice time the residual has not been coded. What is
/// compared is signalling against signalling.
///
/// Each of those is a bounded, named offset shared by every candidate at
/// the same CU, which is what a comparison needs — the shapes are ranked
/// against each other, and a common offset cancels.
pub(crate) struct Rate {
    /// The slice's initial contexts, cloned per pricing.
    cx: Contexts,
    /// log2 of the CU size, for `inter_pred_idc`'s block dimensions and
    /// for whether `split_cu_flag` is coded at all (not at the 8x8
    /// minimum coding block).
    log2_cu: u32,
    /// The CU's coding-tree depth, `CtDepth` — the context of
    /// `inter_pred_idc`'s first bin.
    depth: u32,
}

impl Rate {
    /// `init_type` as `code_inter_picture` derives it: 1 for P, 2 for B.
    #[cfg(test)]
    pub(crate) fn new(qp: i32, is_b: bool, log2_cu: u32) -> Self {
        Self::new_at(qp, is_b, log2_cu, 0)
    }

    /// The rate model for a `1 << log2_cu` CU at coding-tree depth
    /// `depth` of a P (`is_b` false) or B slice at quantiser `qp`.
    pub(crate) fn new_at(qp: i32, is_b: bool, log2_cu: u32, depth: u32) -> Self {
        Rate {
            cx: crate::encode::h265::initial_contexts(if is_b { 2 } else { 1 }, qp),
            log2_cu,
            depth,
        }
    }

    /// Run `f` over a counting encoder and a private copy of the contexts.
    /// Fractional bits, not emitted ones: a shape short enough to fit
    /// inside the arithmetic coder's first output byte emits nothing at
    /// all, so `bits_counted` would price several distinct shapes at zero
    /// and delete the rate term from the comparison. See
    /// `CabacEncoder::fractional_bits`.
    fn count(&self, f: impl FnOnce(&mut CabacEncoder<'static>, &mut Contexts)) -> f32 {
        let mut cx = self.cx.clone();
        let mut e = CabacEncoder::estimating();
        f(&mut e, &mut cx);
        e.fractional_bits() as f32
    }

    /// The elements every inter CU spells before its shape diverges:
    /// `split_cu_flag` then `cu_skip_flag`. `nb` is the neutral neighbour
    /// context described on [`Rate`].
    fn prefix(&self, e: &mut CabacEncoder<'static>, cx: &mut Contexts, skip: bool) {
        // `split_cu_flag` exists only above the 8x8 minimum coding block;
        // at it the reader infers the leaf and takes no bin.
        if self.log2_cu > 3 {
            let nb = SplitCuNb {
                left_depth: None,
                above_depth: None,
            };
            write_split_cu_flag(e, cx, &nb, self.depth, false);
        }
        write_cu_skip_flag(e, cx, None, None, skip);
    }

    /// `cu_skip_flag` 1 and a `merge_idx`; the reader infers the rest.
    pub(crate) fn skip(&self, merge_idx: u8) -> f32 {
        self.count(|e, cx| {
            self.prefix(e, cx, true);
            write_merge_idx(e, cx, MAX_MERGE_CAND as u32, u32::from(merge_idx));
        })
    }

    /// A non-skip 2Nx2N merge CU. `rqt_root_cbf` is not coded — the reader
    /// infers it — so nothing stands in for it here either.
    pub(crate) fn merge(&self, merge_idx: u8) -> f32 {
        self.count(|e, cx| {
            self.prefix(e, cx, false);
            write_pred_mode_flag(e, cx, false);
            write_part_mode_inter(e, cx, PartMode::P2Nx2N);
            write_merge_flag(e, cx, true);
            write_merge_idx(e, cx, MAX_MERGE_CAND as u32, u32::from(merge_idx));
        })
    }

    /// P-slice AMVP at one reference: one list, its `mvd` and
    /// `mvp_l0_flag`, then `rqt_root_cbf` — whose value is a parameter
    /// because it is a coded bin with a cost, and because it is what lets
    /// a test price exactly the shape `write_cu_inter` emits.
    /// [`Rate::amvp_ref`] with a single active reference, which spells no
    /// `ref_idx` — kept for the test that prices the shape by hand.
    #[cfg(test)]
    pub(crate) fn amvp(&self, mvd: Mv, mvp_flag: u8, root_cbf: bool) -> f32 {
        self.amvp_ref(mvd, mvp_flag, root_cbf, 1, 0)
    }

    /// [`Rate::amvp`] plus what it costs to name the reference: `ref_idx`
    /// is truncated unary over the active count `nref`, and absent
    /// entirely when the list has one entry — so at one reference this
    /// prices identically to a stream that never had the choice, which
    /// is what keeps single-reference streams byte-identical.
    pub(crate) fn amvp_ref(
        &self,
        mvd: Mv,
        mvp_flag: u8,
        root_cbf: bool,
        nref: u32,
        ref_idx: u32,
    ) -> f32 {
        self.count(|e, cx| {
            self.prefix(e, cx, false);
            write_pred_mode_flag(e, cx, false);
            write_part_mode_inter(e, cx, PartMode::P2Nx2N);
            write_merge_flag(e, cx, false);
            if nref > 1 {
                write_ref_idx(e, cx, nref, ref_idx);
            }
            write_mvd(e, cx, mvd);
            write_mvp_flag(e, cx, mvp_flag != 0);
            write_rqt_root_cbf(e, cx, root_cbf);
        })
    }

    /// What a partitioned CU spells around its two prediction units:
    /// `split_cu_flag` and `cu_skip_flag` 0 (the prefix), `pred_mode_flag`,
    /// `part_mode` and `rqt_root_cbf` 1. The units' own syntax is
    /// [`Rate::pu`], so a shape is priced as this plus its two units.
    pub(crate) fn parts(&self, part: PartMode) -> f32 {
        self.count(|e, cx| {
            self.prefix(e, cx, false);
            write_pred_mode_flag(e, cx, false);
            write_part_mode_inter_at(e, cx, part, self.log2_cu, MIN_CB_LOG2, false);
            write_rqt_root_cbf(e, cx, true);
        })
    }

    /// One prediction unit of a partitioned CU: `merge_flag`, then its
    /// `merge_idx` or its AMVP syntax, exactly as `write_prediction_unit`
    /// spells it for the unit `pu`, in a B slice or a P slice of `nref`
    /// active list-0 references.
    pub(crate) fn pu(&self, sig: &PuPick, is_b: bool, pu: &PuPos, nref: u32) -> f32 {
        self.count(|e, cx| write_prediction_unit(e, cx, sig, is_b, pu.w, pu.h, self.depth, nref))
    }

    /// B-slice AMVP: `inter_pred_idc`, then per list — interleaved as
    /// `prediction_unit` reads them — the `mvd` and `mvp_lX_flag` of each
    /// list the shape uses.
    pub(crate) fn amvp_b(&self, idc: u8, mvd: [Mv; 2], mvp_flag: [u8; 2], root_cbf: bool) -> f32 {
        let n = 1i32 << self.log2_cu;
        self.count(|e, cx| {
            self.prefix(e, cx, false);
            write_pred_mode_flag(e, cx, false);
            write_part_mode_inter(e, cx, PartMode::P2Nx2N);
            write_merge_flag(e, cx, false);
            write_inter_pred_idc(e, cx, n, n, self.depth, u32::from(idc));
            for list in 0..2usize {
                let uses = match idc {
                    0 => list == 0,
                    1 => list == 1,
                    _ => true,
                };
                if !uses {
                    continue;
                }
                write_mvd(e, cx, mvd[list]);
                write_mvp_flag(e, cx, mvp_flag[list] != 0);
            }
            write_rqt_root_cbf(e, cx, root_cbf);
        })
    }
}

/// Spell one prediction unit of a partitioned CU in `prediction_unit`'s
/// order: `merge_flag`; then `merge_idx`, or `inter_pred_idc` (B slices
/// only, for the `w` by `h` unit at coding-tree depth `depth`) and, for each
/// list the unit uses, `ref_idx` (only where the list has more than one
/// active reference, which here is list 0 of a P slice with `nref` above
/// one), `mvd` and `mvp_lX_flag`. A 2Nx2N CU's single unit is spelled by the
/// CU writer, whose shapes (skip, and the inferred `rqt_root_cbf` of a
/// merge) exist only there.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_prediction_unit(
    e: &mut CabacEncoder,
    cx: &mut Contexts,
    pu: &PuPick,
    is_b: bool,
    w: i32,
    h: i32,
    depth: u32,
    nref: u32,
) {
    if let Some(idx) = pu.merge_idx {
        write_merge_flag(e, cx, true);
        write_merge_idx(e, cx, MAX_MERGE_CAND as u32, u32::from(idx));
        return;
    }
    write_merge_flag(e, cx, false);
    debug_assert!(is_b || pu.idc == 0, "a P unit predicts from list 0");
    debug_assert!(
        pu.idc != 2 || w + h != 12,
        "an 8x4 or 4x8 unit has no PRED_BI"
    );
    if is_b {
        write_inter_pred_idc(e, cx, w, h, depth, u32::from(pu.idc));
    }
    for list in 0..2usize {
        let uses = match pu.idc {
            0 => list == 0,
            1 => list == 1,
            _ => true,
        };
        if !uses {
            continue;
        }
        if list == 0 && !is_b && nref > 1 {
            write_ref_idx(e, cx, nref, u32::from(pu.ref_idx[0].max(0) as u8));
        }
        write_mvd(e, cx, pu.mvd[list]);
        write_mvp_flag(e, cx, pu.mvp_flag[list] != 0);
    }
}

/// Whether to hand this CU to the intra decision: the H.264 module's
/// flatness proxy (`super::h264_me::placeholder_inter_or_intra`) at CU
/// size and generic sample width — a DC prediction costing one SATD and no
/// reconstruction state, with all of that function's stated limits.
///
/// # It carries no rate term, and measurement says that is fine
///
/// Every other decision in this encoder now prices its candidates in real
/// bits. This one does not, and the obvious next step — give it a rate
/// term like the rest — was measured before being written, and would
/// change nothing. Over 808 CU decisions (three clips, two quantisers,
/// the encoder's own CU size rather than a chosen one):
///
/// - intra wins 1.0% of the time;
/// - **0.0% of decisions fall within 10%** of the boundary, and 0.2%
///   within 25%;
/// - the median separation is 1.22, meaning the two sides typically
///   differ by more than the whole inter SATD.
///
/// A Lagrangian rate term moves a comparison by lambda times a bit
/// difference. Nothing that size flips a decision separated by more than
/// 100%, so adding one would cost work and change no output.
///
/// # What that probe does NOT bound, stated because it is the same trap
///
/// It measures how marginal the comparison *as written* is, so it bounds
/// the effect of adding a term to that comparison. It says nothing about
/// replacing the DC-flat distortion proxy, because a bad proxy can sit
/// far from the boundary and still be on the wrong side of it — being
/// unmarginal is not being right. If this decision is ever worth
/// improving, the proxy is the thing to attack, and testing that needs a
/// different probe: code both candidates properly and compare real costs,
/// rather than asking whether the existing numbers are close.
pub fn prefer_intra<S: Sample>(
    ctx: &MeCtx<'_, S>,
    inter_satd: u32,
    src: &[S],
    src_stride: usize,
    n: usize,
) -> bool {
    let mut sum = 0u64;
    for y in 0..n {
        for x in 0..n {
            sum += src[y * src_stride + x].to_i32() as u64;
        }
    }
    let dc = S::from_i32(((sum + (n * n / 2) as u64) / (n * n) as u64) as i32);
    let flat = vec![dc; n * n];
    let intra_proxy = (ctx.dist.satd)(src, src_stride, &flat, n, n, n);
    intra_proxy < inter_satd
}

/// Forward-code and reconstruct one inter transform block whose
/// *prediction is already in the plane*: residual against `src`, forward
/// DCT (inter TUs never take the DST, 8.6.4.2) and quantisation with the
/// inter rounding offset, then reconstruction through the decoder's own
/// `scale_coefficients`, inverse transform and `add_residual`. This is
/// `h265_intra::code_residual` minus its intra-only branches — copied, and
/// flagged in the module header.
#[allow(clippy::too_many_arguments)]
fn code_residual_inter<S: Sample>(
    ctx: &MeCtx<'_, S>,
    plane: &mut Plane16<S>,
    x: usize,
    y: usize,
    log2: u32,
    c_idx: usize,
    qp: i32,
    src: &[S],
    src_stride: usize,
    levels: &mut [i16],
) -> u32 {
    // The scratch sized to the block: zeroing a 32x32 one for every 4x4
    // block was a measurable share of an encode.
    match log2 {
        2 => code_residual_inter_n::<S, 16>(
            ctx, plane, x, y, log2, c_idx, qp, src, src_stride, levels,
        ),
        3 => code_residual_inter_n::<S, 64>(
            ctx, plane, x, y, log2, c_idx, qp, src, src_stride, levels,
        ),
        4 => code_residual_inter_n::<S, 256>(
            ctx, plane, x, y, log2, c_idx, qp, src, src_stride, levels,
        ),
        _ => code_residual_inter_n::<S, 1024>(
            ctx, plane, x, y, log2, c_idx, qp, src, src_stride, levels,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn code_residual_inter_n<S: Sample, const N: usize>(
    ctx: &MeCtx<'_, S>,
    plane: &mut Plane16<S>,
    x: usize,
    y: usize,
    log2: u32,
    c_idx: usize,
    qp: i32,
    src: &[S],
    src_stride: usize,
    levels: &mut [i16],
) -> u32 {
    let n = 1usize << log2;
    debug_assert_eq!(n * n, N);
    let off = plane.offset(x as isize, y as isize);
    let stride = plane.stride;
    let max = (1i32 << ctx.bit_depth) - 1;

    let mut work = [0i16; N];
    for yy in 0..n {
        for xx in 0..n {
            work[yy * n + xx] = (src[yy * src_stride + xx].to_i32()
                - plane.data[off + yy * stride + xx].to_i32())
                as i16;
        }
    }

    if ctx.bypass {
        // Lossless: the residual IS the coefficients. `residual_block`
        // skips `scale_coefficients` and the inverse transform for a
        // bypassed CU (`hevc::ctu`'s `if !cu.bypass` gate) and adds what
        // it parsed as it stands, so carrying the raw difference makes
        // prediction plus residual equal the source — and the clip inside
        // `add_residual` never bites, because that sum is a real sample
        // value by construction. The same branch `h265_intra`'s
        // `code_residual` takes, and for the same reason.
        levels[..n * n].copy_from_slice(&work[..n * n]);
        (ctx.dsp.add_residual)(&mut plane.data[off..], stride, &work, n, max);
        return levels[..n * n].iter().filter(|&&v| v != 0).count() as u32;
    }

    (ctx.enc.fdct[(log2 - 2) as usize])(&mut work, log2, ctx.bit_depth);
    let qb = qbits(qp, log2, ctx.bit_depth);
    let nz = match ctx.rdoq {
        Some(cx) => super::h265_rdoq::quantise(
            &super::h265_rdoq::Block {
                log2,
                c_idx,
                scan_idx: 0,
                qp,
                bit_depth: ctx.bit_depth,
                lambda: ssd_lambda(ctx.qp, ctx.bit_depth),
                cx,
                sign_hiding: ctx.sign_hiding,
            },
            &work,
            levels,
        ),
        None => {
            let nz = (ctx.enc.quant)(
                &work,
                levels,
                n,
                quant_scale((qp % 6) as usize),
                qb,
                quant_offset(qb, false),
            );
            if ctx.sign_hiding && nz > 0 {
                super::h265_rdoq::hide_signs(
                    &super::h265_rdoq::Block {
                        log2,
                        c_idx,
                        scan_idx: 0,
                        qp,
                        bit_depth: ctx.bit_depth,
                        lambda: 0.0,
                        cx: ctx.rdoq.unwrap_or(super::h265_intra::no_contexts()),
                        sign_hiding: true,
                    },
                    &work,
                    levels,
                )
            } else {
                nz
            }
        }
    };

    if nz == 0 {
        // Every level is zero: the decoder adds nothing to the prediction.
        return 0;
    }
    work[..n * n].copy_from_slice(&levels[..n * n]);
    scale_coefficients(
        &mut work,
        log2,
        qp,
        ctx.bit_depth,
        ScalingSource::Flat,
        false,
        n - 1,
        n - 1,
    );
    let bd_shift = 20 - ctx.bit_depth as i32;
    (ctx.dsp.idct[(log2 - 2) as usize])(&mut work, bd_shift, n - 1, n - 1);
    (ctx.dsp.add_residual)(&mut plane.data[off..], stride, &work, n, max);
    nz
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::Cpu;
    use crate::dsp::distortion::DistortionDsp;
    use crate::dsp::hevc::HevcDsp;
    use crate::dsp::hevc_enc::HevcEncDsp;
    use crate::encode::Config;
    use crate::encode::h265_syntax::{Geometry as SynGeometry, write_pps, write_sps};
    use crate::hevc::mvpred::merge_candidate;
    use crate::picture::ChromaFormat;

    fn lcg(s: &mut u64) -> u32 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*s >> 33) as u32
    }

    struct Kit {
        dsp: HevcDsp<u8>,
        enc: HevcEncDsp,
        dist: DistortionDsp<u8>,
    }

    impl Kit {
        fn new() -> Self {
            Kit {
                dsp: HevcDsp::new(Cpu::SCALAR),
                enc: HevcEncDsp::scalar(),
                dist: DistortionDsp::scalar(),
            }
        }
        fn ctx(&self, qp: i32) -> MeCtx<'_, u8> {
            IntraCtx {
                dsp: &self.dsp,
                enc: &self.enc,
                dist: &self.dist,
                qp,
                bit_depth: 8,
                strong_smoothing: false,
                bypass: false,
                free_to_trim: false,
                sign_hiding: false,
                rdoq: None,
                fast_intra: false,
            }
        }
    }

    /// The parameter sets this stream would carry, through the decoder's
    /// own parsers — the same round trip the encoder applies to everything
    /// it writes.
    fn parsed_sets(w: u32, h: u32) -> (Sps, Pps) {
        parsed_sets_fmt(w, h, ChromaFormat::Yuv420)
    }

    /// The same, in a chosen chroma format — `chroma_format_idc` travels
    /// through `write_sps` and back out of the decoder's parser, so the
    /// decision module reads the format from a round trip rather than from
    /// the caller's word for it.
    fn parsed_sets_fmt(w: u32, h: u32, chroma: ChromaFormat) -> (Sps, Pps) {
        let cfg = Config {
            width: w,
            height: h,
            gop: 8,
            chroma,
            ..Config::default()
        };
        let syn = SynGeometry::new(&cfg);
        let sps = Sps::parse(&crate::nal::unescape_rbsp(&write_sps(&cfg, &syn, 16, None))).unwrap();
        let mut pps = Pps::parse(&crate::nal::unescape_rbsp(&write_pps(26, false, false))).unwrap();
        pps.resolve_tiles(&sps).unwrap();
        (sps, pps)
    }

    /// A reference picture whose luma is the H.264 module's triangle
    /// grating (periods 25 and 27) — see `grating_plane` in
    /// `h264_me.rs`'s tests for the three failure modes that shaped it.
    /// The constraint carries over unchanged: a greedy diamond seeded at
    /// zero converges to the true offset only when the walk starts inside
    /// the true basin (half-period, ≈ 12) and no alias is nearer, so the
    /// zero-seeded assertions below keep |d| small; larger motion is
    /// found the way a picture walk finds it, by neighbour propagation
    /// through the merge candidates. Chroma is never scored by the
    /// search, so radial bowls do. Borders extended as the decoder
    /// extends them before MC reads.
    fn reference(w: usize, h: usize, seed: u64) -> Frame<u8> {
        reference_fmt(w, h, seed, ChromaFormat::Yuv420)
    }

    /// The same reference in a chosen chroma format. The chroma bowls are
    /// laid out over whatever plane the format gives, so 4:2:2's
    /// full-height and 4:4:4's full-size chroma each carry real structure
    /// rather than a stretched copy of the 4:2:0 one — content that a
    /// wrongly scaled chroma vector would visibly miss.
    fn reference_fmt(w: usize, h: usize, seed: u64, chroma: ChromaFormat) -> Frame<u8> {
        let mut f = Frame::new(w, h, chroma, 8);
        f.poc = 0;
        let _ = seed;
        for y in 0..h {
            for x in 0..w {
                let tx = (x as i32 % 25 - 12).abs();
                let ty = (y as i32 % 27 - 13).abs();
                let off = f.y.offset(x as isize, y as isize);
                f.y.data[off] = (40 + 4 * tx + 3 * ty) as u8;
            }
        }
        let (cw, ch) = (f.cb.width, f.cb.height);
        for y in 0..ch {
            for x in 0..cw {
                let r2 = (x as i32 - (cw / 2) as i32).pow(2) + (y as i32 - (ch / 2) as i32).pow(2);
                let off = f.cb.offset(x as isize, y as isize);
                f.cb.data[off] = (200 - r2.min(160)) as u8;
                let off = f.cr.offset(x as isize, y as isize);
                f.cr.data[off] = (60 + r2.min(160)) as u8;
            }
        }
        f.extend_rows(0, h);
        f
    }

    /// Source planes translated by `(dx, dy)` full luma samples relative
    /// to `refp` (a block at `p` in the source equals the reference at
    /// `p + (dx, dy)`), reads outside the picture clamped as the padded
    /// plane clamps them.
    fn translated(refp: &Frame<u8>, dx: i32, dy: i32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let (w, h) = (refp.width, refp.height);
        let mut y = vec![0u8; w * h];
        for yy in 0..h {
            for xx in 0..w {
                y[yy * w + xx] = refp.y.at_clamped(xx as i32 + dx, yy as i32 + dy);
            }
        }
        // The chroma shift is the luma one divided by (SubWidthC,
        // SubHeightC): a 4:2:2 chroma plane is not subsampled vertically,
        // so its rows shift by the *whole* `dy`, and a caller wanting an
        // integral chroma translation must therefore keep `dx` even but
        // may leave `dy` odd. Taking this from the frame's own format is
        // what makes the fixture agree with `predict_block`'s `mvc`.
        let (sw, sh) = refp.chroma.subsampling();
        let (sw, sh) = (sw as i32, sh as i32);
        let (cw, ch) = (refp.cb.width, refp.cb.height);
        let mut cb = vec![0u8; cw * ch];
        let mut cr = vec![0u8; cw * ch];
        for yy in 0..ch {
            for xx in 0..cw {
                cb[yy * cw + xx] = refp.cb.at_clamped(xx as i32 + dx / sw, yy as i32 + dy / sh);
                cr[yy * cw + xx] = refp.cr.at_clamped(xx as i32 + dx / sw, yy as i32 + dy / sh);
            }
        }
        (y, cb, cr)
    }

    /// The average of two references, each shifted by its own full-sample
    /// vector: the content a B picture between them carries. `d0` is the
    /// list-0 motion and `d1` the list-1 motion, in luma samples; chroma
    /// shifts by the same vector divided by this format's (SubWidthC,
    /// SubHeightC), so an even vector stays integral in every format.
    /// Monochrome returns empty chroma planes.
    fn bi_translated(
        r0: &Frame<u8>,
        d0: (i32, i32),
        r1: &Frame<u8>,
        d1: (i32, i32),
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let (w, h) = (r0.width, r0.height);
        let mut y = vec![0u8; w * h];
        for yy in 0..h {
            for xx in 0..w {
                let a = r0.y.at_clamped(xx as i32 + d0.0, yy as i32 + d0.1) as i32;
                let b = r1.y.at_clamped(xx as i32 + d1.0, yy as i32 + d1.1) as i32;
                y[yy * w + xx] = ((a + b + 1) >> 1) as u8;
            }
        }
        let (sw, sh) = r0.chroma.subsampling();
        let (sw, sh) = (sw as i32, sh as i32);
        let (cw, ch) = (r0.cb.width, r0.cb.height);
        let mut cb = vec![0u8; cw * ch];
        let mut cr = vec![0u8; cw * ch];
        for yy in 0..ch {
            for xx in 0..cw {
                for (plane0, plane1, dst) in [(&r0.cb, &r1.cb, &mut cb), (&r0.cr, &r1.cr, &mut cr)]
                {
                    let a = plane0.at_clamped(xx as i32 + d0.0 / sw, yy as i32 + d0.1 / sh) as i32;
                    let b = plane1.at_clamped(xx as i32 + d1.0 / sw, yy as i32 + d1.1 / sh) as i32;
                    dst[yy * cw + xx] = ((a + b + 1) >> 1) as u8;
                }
            }
        }
        (y, cb, cr)
    }

    fn code_picture(
        ctx: &MeCtx<'_, u8>,
        sps: &Sps,
        pps: &Pps,
        refp: &Frame<u8>,
        src_y: &[u8],
        src_cb: &[u8],
        src_cr: &[u8],
    ) -> (InterPicture<u8>, Vec<InterCuDecision>) {
        let (w, h) = (sps.width as usize, sps.height as usize);
        let n = 1usize << sps.log2_ctb_size;
        let (sw, _) = sub_wh(sps.chroma_array_type());
        let c_stride = if sps.chroma_array_type() == 0 {
            0
        } else {
            w / sw
        };
        let mut pic = InterPicture::new(sps, pps, 1);
        let mut decisions = Vec::new();
        for cy in 0..h / n {
            for cx in 0..w / n {
                decisions.push(pic.code_ctu(
                    ctx,
                    &[refp],
                    cx,
                    cy,
                    src_y,
                    w,
                    src_cb,
                    src_cr,
                    c_stride,
                ));
            }
        }
        (pic, decisions)
    }

    /// Every decision's internal consistency: `Skip` carries nothing,
    /// `Merge` always carries residual (the inferred `rqt_root_cbf`), the
    /// cbf flags agree with the coefficient arrays.
    fn assert_invariants(decisions: &[InterCuDecision], cat: u32) {
        for (i, d) in decisions.iter().enumerate() {
            let n = 1usize << d.log2_cu;
            let nz_l = d.luma[..n * n].iter().any(|&v| v != 0);
            // The chroma slots this format uses, at this format's TB size.
            let (_, ntb, log2c) = chroma_tbs(cat, 0, 0, d.log2_cu);
            let nc2 = 1usize << (2 * log2c);
            let slot = |comp: usize, t: usize| {
                d.chroma[comp][t * nc2..(t + 1) * nc2]
                    .iter()
                    .any(|&v| v != 0)
            };
            let mut nz_c = false;
            for comp in 0..2 {
                assert_eq!(
                    d.cbf_chroma[comp],
                    ntb > 0 && slot(comp, 0),
                    "cu {i}: cbf_chroma[{comp}] disagrees with the levels"
                );
                assert_eq!(
                    d.cbf_chroma_bot[comp],
                    ntb > 1 && slot(comp, 1),
                    "cu {i}: cbf_chroma_bot[{comp}] disagrees with the levels"
                );
                nz_c |= d.cbf_chroma[comp] || d.cbf_chroma_bot[comp];
            }
            if cat == 0 {
                assert!(!nz_c, "cu {i}: monochrome carries a chroma cbf");
            }
            if cat != 2 {
                assert_eq!(
                    d.cbf_chroma_bot, [false; 2],
                    "cu {i}: only 4:2:2 has a bottom chroma square"
                );
            }
            // Every slot this format does not use must be untouched: the
            // writer indexes by slot, so a stray level there would be
            // spelled into some other format's stream shape.
            for comp in 0..2 {
                let used = ntb * nc2;
                assert!(
                    d.chroma[comp][used..].iter().all(|&v| v == 0),
                    "cu {i}: levels beyond the format's chroma slots"
                );
            }
            assert_eq!(
                d.cbf_luma, nz_l,
                "cu {i}: cbf_luma disagrees with the levels"
            );
            assert_eq!(
                d.rqt_root_cbf,
                nz_l || nz_c,
                "cu {i}: rqt_root_cbf disagrees"
            );
            // The inference trap the writer relies on: at an inter leaf of
            // depth 0 with no chroma cbf the reader reads no cbf_luma bin
            // and infers 1, so a coded tree with neither is unspellable.
            if d.rqt_root_cbf && !nz_c {
                assert!(
                    d.cbf_luma,
                    "cu {i}: a coded tree with no chroma cbf must have cbf_luma 1 (the reader infers it)"
                );
            }
            match d.kind {
                InterCuKind::Skip { .. } => {
                    assert!(!d.rqt_root_cbf, "cu {i}: a skip CU carries residual")
                }
                InterCuKind::Merge { .. } => {
                    // The reader infers rqt_root_cbf 1 for a non-skip
                    // 2Nx2N merge CU: producing one without residual
                    // would desync — such a CU must have been Skip.
                    assert!(
                        d.rqt_root_cbf,
                        "cu {i}: a zero-residual merge CU escaped becoming Skip"
                    )
                }
                // A partitioned CU codes rqt_root_cbf, so either value is
                // spellable.
                InterCuKind::Amvp { .. }
                | InterCuKind::BAmvp { .. }
                | InterCuKind::Parts
                | InterCuKind::UseIntra => {}
            }
        }
    }

    #[test]
    fn an_integral_translation_is_found_exactly_and_skips_when_it_can() {
        let kit = Kit::new();
        let ctx = kit.ctx(26);
        let (sps, pps) = parsed_sets(64, 32);
        let refp = reference(64, 32, 7);
        // Even components, so the chroma offset (dx/2, dy/2) is integral
        // too and the prediction is a plain copy in all three planes — an
        // odd luma translation is a *half-sample* chroma one, which the
        // integer-shifted chroma source of `translated` cannot match.
        let (dx, dy) = (-6i32, 4i32);
        let (sy, scb, scr) = translated(&refp, dx, dy);
        let (pic, decisions) = code_picture(&ctx, &sps, &pps, &refp, &sy, &scb, &scr);
        assert_invariants(&decisions, 1);
        let want = Mv::new((dx * 4) as i16, (dy * 4) as i16);
        for (i, d) in decisions.iter().enumerate() {
            assert_eq!(d.mv, want, "cu {i} missed the translation: {:?}", d.kind);
            assert!(
                !d.rqt_root_cbf,
                "cu {i}: an exact translation left residual"
            );
        }
        // The first CU has no motion neighbours, so its candidates are the
        // zero-vector pads: it signals AMVP with no residual. Every later
        // CU sees the translation in a spatial candidate and skips.
        assert!(
            matches!(decisions[0].kind, InterCuKind::Amvp { .. }),
            "first CU: {:?}",
            decisions[0].kind
        );
        for (i, d) in decisions.iter().enumerate().skip(1) {
            assert!(
                matches!(d.kind, InterCuKind::Skip { .. }),
                "cu {i}: {:?}",
                d.kind
            );
        }
        // And the reconstruction is the translated reference, exactly: the
        // prediction was the decoder's own and no residual was added.
        for y in 0..32usize {
            for x in 0..64usize {
                let off = pic.recon.y.offset(x as isize, y as isize);
                assert_eq!(pic.recon.y.data[off], sy[y * 64 + x], "luma ({x},{y})");
            }
        }
        for y in 0..16usize {
            for x in 0..32usize {
                let off = pic.recon.cb.offset(x as isize, y as isize);
                assert_eq!(pic.recon.cb.data[off], scb[y * 32 + x], "cb ({x},{y})");
                let off = pic.recon.cr.offset(x as isize, y as isize);
                assert_eq!(pic.recon.cr.data[off], scr[y * 32 + x], "cr ({x},{y})");
            }
        }
    }

    #[test]
    fn a_surviving_residual_forbids_skip() {
        let kit = Kit::new();
        let ctx = kit.ctx(26);
        let (sps, pps) = parsed_sets(64, 32);
        let refp = reference(64, 32, 11);
        let (mut sy, scb, scr) = translated(&refp, 2, -1);
        // Structured luma damage well above what QP 26 quantises away.
        let mut s = 99u64;
        for v in sy.iter_mut() {
            let d = (lcg(&mut s) % 64) as i32 - 32;
            *v = (*v as i32 + d).clamp(0, 255) as u8;
        }
        let (_, decisions) = code_picture(&ctx, &sps, &pps, &refp, &sy, &scb, &scr);
        assert_invariants(&decisions, 1);
        assert!(
            decisions.iter().any(|d| d.rqt_root_cbf),
            "no CU carried residual — the damage was supposed to survive quantisation"
        );
        for (i, d) in decisions.iter().enumerate() {
            if d.rqt_root_cbf {
                assert!(
                    !matches!(d.kind, InterCuKind::Skip { .. }),
                    "cu {i} skipped with residual"
                );
            }
        }
    }

    #[test]
    fn a_half_sample_translation_is_found_by_the_refinement() {
        let kit = Kit::new();
        let ctx = kit.ctx(26);
        let (sps, pps) = parsed_sets(64, 32);
        let refp = reference(64, 32, 13);
        // The source is the decoder's own half-sample interpolation of the
        // reference: predict every CTU of a scratch frame at mv (2, 0).
        let want = Mv::new(2, 0);
        let mut interp = Frame::<u8>::new(64, 32, ChromaFormat::Yuv420, 8);
        let mut scratch = McScratch::new();
        for cy in 0..2usize {
            for cx in 0..4usize {
                predict_block(
                    &kit.dsp,
                    &mut scratch,
                    &mut interp,
                    cx * 16,
                    cy * 16,
                    16,
                    16,
                    Some((&refp, want)),
                    None,
                    [Weighting::Default; 3],
                );
            }
        }
        let flat = |p: &Plane16<u8>, w: usize, h: usize| -> Vec<u8> {
            let mut v = vec![0u8; w * h];
            for y in 0..h {
                for x in 0..w {
                    v[y * w + x] = p.data[p.offset(x as isize, y as isize)];
                }
            }
            v
        };
        let sy = flat(&interp.y, 64, 32);
        let scb = flat(&interp.cb, 32, 16);
        let scr = flat(&interp.cr, 32, 16);
        let (_, decisions) = code_picture(&ctx, &sps, &pps, &refp, &sy, &scb, &scr);
        assert_invariants(&decisions, 1);
        for (i, d) in decisions.iter().enumerate() {
            assert_eq!(d.mv, want, "cu {i}: {:?}", d.kind);
            assert!(
                !d.rqt_root_cbf,
                "cu {i}: the exact interpolation left residual"
            );
        }
    }

    /// The B anchor: every B decision replayed through the decoder's own
    /// candidate derivation over an independently maintained state, once
    /// per chroma format.
    ///
    /// This is [`every_decision_replays_through_an_independent_decoder_state`]
    /// for the two-list case, and it proves the same thing plus what only
    /// B has: that `merge_candidate` with `is_b` re-derives the *pair* of
    /// vectors and reference indices the decision recorded, that a
    /// per-list AMVP replays through `amvp(list)` and the wrapping mvd
    /// sum, and that the bi reconstruction the encoder holds is the one
    /// `predict_block` builds from both lists.
    ///
    /// The vacuity guards matter as much as the replay: content that never
    /// chose BI, or never chose merge, would let a broken second list pass
    /// unnoticed, so both are asserted to have occurred.
    #[test]
    fn every_b_decision_replays_through_an_independent_decoder_state() {
        for chroma in [
            ChromaFormat::Monochrome,
            ChromaFormat::Yuv420,
            ChromaFormat::Yuv422,
            ChromaFormat::Yuv444,
        ] {
            // Three sources per format, because ONE source does not reach
            // the shapes. An earlier version of this test used only the
            // bi fixture and every CU came out `Merge` at the zero vector;
            // `BAmvp` never occurred at all, and a mutation sweep showed
            // five of six seeded faults passing. The scenarios exist to
            // drive each `inter_pred_idc` for real.
            let mut seen_idc = [false; 3];
            for scen in [BScenario::Uni0, BScenario::Uni1, BScenario::Bi] {
                replay_b_one_format(chroma, scen, &mut seen_idc, None);
            }
            assert!(
                seen_idc[0],
                "{chroma:?}: no CU ever coded PRED_L0 through AMVP"
            );
            assert!(
                seen_idc[1],
                "{chroma:?}: no CU ever coded PRED_L1 through AMVP"
            );
            // PRED_BI through AMVP is deliberately NOT asserted, and the
            // reason is a property of the decision rather than a gap in
            // this test. Whenever both lists find good vectors, the merge
            // list already holds an equivalent two-list candidate — from a
            // neighbour, a combined bi-predictive pair, or the bi zero
            // candidate — and merge costs a couple of bins against AMVP-BI's
            // `inter_pred_idc` plus two mvds plus two mvp flags. Merge
            // therefore wins on rate, correctly. Bi prediction and bi
            // reconstruction ARE exercised here, through those merge CUs
            // (the `BScenario::Bi` guard below requires a coded two-list
            // CU); what is not exercised is the AMVP-BI *signalling*.
            //
            // That signalling is covered where it can be driven directly:
            // `hevc::ctu`'s `b_inter_pred_idc_round_trips_by_value` writes
            // all three `inter_pred_idc` values through the production
            // writer and reads them back with the production decoder. If
            // an iterative bi refinement ever lands, AMVP-BI should start
            // winning here and this comment becomes an assertion.
            let _ = seen_idc[2];
        }
    }

    /// The B replay under explicit weighting: the same three scenarios,
    /// with each anchor handed to the decision as the picture its list's
    /// table entry weights back to the content the source was built from
    /// — so the weighted prediction, not the default one, is what matches.
    /// The list-0 and list-1 entries differ in every component, gain and
    /// offset, luma and chroma, so a walk that predicted without the
    /// weights, or weighted one list with the other's entry, diverges from
    /// the replay, which derives its weighting from the table through the
    /// reader's own `explicit_weighting`.
    ///
    /// The second table carries offsets alone, large and of opposite signs.
    /// A one-list search scored without the weighting then sees each
    /// candidate a full offset from the source in every sample, and the
    /// AMVP one-list shapes lose to the weighted merge and bi candidates
    /// that score as the prediction will be made — so the PRED_L0 / PRED_L1
    /// guards fail on a refinement that forgot the weights, which the gentle
    /// table alone let pass. The table is usable at all only because the
    /// whole-sample search scores on the weighted reference
    /// (`InterPicture::search_b`): searched on the reference itself, an
    /// offset this size flattens the grating's SAD landscape and the
    /// unmutated walk misses the one-list shapes too, while an offset of 12
    /// leaves the search working and the two scorings choosing alike.
    #[test]
    fn every_weighted_b_decision_replays_through_an_independent_decoder_state() {
        for (name, table) in [
            ("gain and offset", weighted_b_table()),
            ("offsets", offset_b_table()),
        ] {
            for chroma in [
                ChromaFormat::Monochrome,
                ChromaFormat::Yuv420,
                ChromaFormat::Yuv422,
                ChromaFormat::Yuv444,
            ] {
                let mut seen_idc = [false; 3];
                for scen in [BScenario::Uni0, BScenario::Uni1, BScenario::Bi] {
                    replay_b_one_format(chroma, scen, &mut seen_idc, Some(&table));
                }
                assert!(
                    seen_idc[0],
                    "{chroma:?} weighted ({name}): no CU ever coded PRED_L0 through AMVP"
                );
                assert!(
                    seen_idc[1],
                    "{chroma:?} weighted ({name}): no CU ever coded PRED_L1 through AMVP"
                );
            }
        }
    }

    /// A B slice's table of offsets only, large and of opposite signs per
    /// list (+24 and -24 in luma, smaller in chroma), at unit gain. The
    /// fixture's luma lies in 28..=139, so the inverse (`unweight`) never
    /// clips and the weighted prediction reproduces the content.
    fn offset_b_table() -> crate::hevc::slice::PredWeightTable {
        use crate::hevc::slice::{PredWeightTable, WeightEntry};
        PredWeightTable {
            luma_log2_denom: 6,
            chroma_log2_denom: 6,
            lists: [
                vec![WeightEntry {
                    luma: (64, 24),
                    chroma: [(64, 8), (64, 8)],
                }],
                vec![WeightEntry {
                    luma: (64, -24),
                    chroma: [(64, -8), (64, -8)],
                }],
            ],
        }
    }

    /// A B slice's table whose two lists differ in every component: list 0
    /// a gain below the identity with positive offsets, list 1 a gain above
    /// it with negative ones. Denominators 6, 8-bit offsets.
    fn weighted_b_table() -> crate::hevc::slice::PredWeightTable {
        use crate::hevc::slice::{PredWeightTable, WeightEntry};
        PredWeightTable {
            luma_log2_denom: 6,
            chroma_log2_denom: 6,
            lists: [
                vec![WeightEntry {
                    luma: (56, 12),
                    chroma: [(60, 6), (64, 3)],
                }],
                vec![WeightEntry {
                    luma: (72, -10),
                    chroma: [(66, -2), (68, -6)],
                }],
            ],
        }
    }

    /// The inverse of the weighting `(w, o)` at denominator 6 over the
    /// picture area of `plane`: each sample `p` becomes the `q` whose
    /// `((q * w + 32) >> 6) + o` lands nearest `p`, so a prediction from
    /// the result under that weighting reproduces the plane it was made
    /// from, to a rounding.
    fn unweight(plane: &mut Plane16<u8>, (w, o): (i32, i32)) {
        for y in 0..plane.height {
            for x in 0..plane.width {
                let off = plane.offset(x as isize, y as isize);
                let p = i32::from(plane.data[off]);
                plane.data[off] = (((p - o) * 64 + w / 2) / w).clamp(0, 255) as u8;
            }
        }
    }

    /// What the source of a B replay looks like, and therefore which
    /// signalling the decision should reach for.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum BScenario {
        /// An exact translation of the list-0 anchor: `PRED_L0` should win.
        Uni0,
        /// An exact translation of the list-1 anchor: `PRED_L1` should win.
        Uni1,
        /// The average of both anchors, each moved toward this picture —
        /// what default-weighted bi-prediction produces, so `PRED_BI`
        /// should win.
        Bi,
    }

    fn replay_b_one_format(
        chroma: ChromaFormat,
        scen: BScenario,
        seen_idc: &mut [bool; 3],
        weights: Option<&crate::hevc::slice::PredWeightTable>,
    ) {
        let kit = Kit::new();
        let ctx = kit.ctx(30);
        let (sps, pps) = parsed_sets_fmt(64, 64, chroma);
        let cat = sps.chroma_array_type();
        // Two anchors around the current picture: POC 1 in the past, POC 3
        // in the future, current POC 2 — a real B, so NoBackwardPredFlag
        // is false and both lists are live.
        // The two anchors must carry DIFFERENT texture, not the same
        // grating at an offset. With identical structure, a per-list
        // search cannot tell which anchor it is looking at: both lists
        // converge on the same vector, their average equals either half,
        // and PRED_BI can never beat a uni shape on rate — so the bi path
        // would go untested however the source were built. Independent
        // noise per anchor is what makes each list's own vector findable
        // and makes averaging genuinely better than either half, which is
        // the whole premise of bi-prediction.
        let mut ref0 = reference_fmt(64, 64, 17, chroma);
        let mut ref1 = reference_fmt(64, 64, 23, chroma);
        for (f, mut seed) in [(&mut ref0, 0x51ed_u64), (&mut ref1, 0xb0a7_u64)] {
            for y in 0..64usize {
                for x in 0..64usize {
                    let off = f.y.offset(x as isize, y as isize);
                    let d = (lcg(&mut seed) % 24) as i32 - 12;
                    f.y.data[off] = (f.y.data[off] as i32 + d).clamp(0, 255) as u8;
                }
            }
        }
        ref0.poc = 1;
        ref0.extend_rows(0, 64);
        ref1.poc = 3;
        ref1.extend_rows(0, 64);

        // The source this scenario asks for. Even vectors throughout, so
        // the chroma translation is integral in every format and the
        // fixture matches what `predict_block` derives.
        let (mut sy, scb, scr) = match scen {
            BScenario::Uni0 => translated(&ref0, 4, 2),
            BScenario::Uni1 => translated(&ref1, -4, -2),
            BScenario::Bi => bi_translated(&ref0, (4, 2), &ref1, (-4, -2)),
        };
        // Light damage, so residual survives and the AMVP shapes are
        // reached rather than everything collapsing to skip. Kept well
        // below the level that would make intra win.
        let mut s = 41u64;
        for yy in 0..64usize {
            for xx in 0..64usize {
                if (xx / 16 + yy / 16) % 3 == 0 {
                    let d = (lcg(&mut s) % 16) as i32 - 8;
                    let v = &mut sy[yy * 64 + xx];
                    *v = (*v as i32 + d).clamp(0, 255) as u8;
                }
            }
        }
        // Under a table, each anchor becomes the picture its own list's
        // entry weights back to the content the source was built from.
        if let Some(t) = weights {
            for (f, e) in [(&mut ref0, &t.lists[0][0]), (&mut ref1, &t.lists[1][0])] {
                unweight(&mut f.y, e.luma);
                unweight(&mut f.cb, e.chroma[0]);
                unweight(&mut f.cr, e.chroma[1]);
                f.extend_rows(0, 64);
            }
        }
        let weighting = |ref_idx: [i8; 2]| {
            weights.map_or([Weighting::Default; 3], |t| {
                crate::hevc::ctu::explicit_weighting(t, 8, 8, ref_idx)
            })
        };

        let (w, h) = (64usize, 64usize);
        let n = 1usize << sps.log2_ctb_size;
        let (sw, _) = sub_wh(cat);
        let c_stride = if cat == 0 { 0 } else { w / sw };
        let mut pic = InterPicture::new(&sps, &pps, 2);
        // The production mapping from the table to each prediction's
        // weighting — so a mapping that bound a list to the other's entry
        // makes one-list prediction lose everywhere and the PRED_L0 /
        // PRED_L1 guards go red; the replay derives its own weighting from
        // each decision's reference indices.
        if let Some(t) = weights {
            pic.set_b_weights(t, 8, 8, &ref0, &ref1);
        }
        let mut decisions = Vec::new();
        for cy in 0..h / n {
            for cx in 0..w / n {
                decisions
                    .push(pic.code_ctu_b(&ctx, &ref0, &ref1, cx, cy, &sy, w, &scb, &scr, c_stride));
            }
        }
        assert_invariants(&decisions, cat);
        let tag = format!(
            "{chroma:?}/{scen:?}{}",
            if weights.is_some() { " weighted" } else { "" }
        );
        // Record which `inter_pred_idc` values were reached, for the
        // caller's aggregate coverage check. Only a CODED CU counts: a
        // `UseIntra` decision still carries the motion fields this module
        // filled before the intra check, so counting those would make the
        // guard vacuous — which is exactly how the earlier version of this
        // test managed to prove nothing.
        for d in &decisions {
            if let InterCuKind::BAmvp { idc, .. } = d.kind {
                seen_idc[idc as usize] = true;
            }
        }
        assert!(
            decisions
                .iter()
                .any(|d| !matches!(d.kind, InterCuKind::UseIntra)),
            "{tag}: every CU went intra; the inter path is untested here"
        );
        // The scenario must actually reach the shape it is named for,
        // through some CU: uni scenarios a single-list CU, the bi scenario
        // a two-list one. Merge candidates count here — they carry lists
        // too — but intra decisions do not.
        let coded = decisions
            .iter()
            .filter(|d| !matches!(d.kind, InterCuKind::UseIntra));
        let hit = match scen {
            BScenario::Uni0 => coded.clone().any(|d| d.ref_idx >= 0 && d.ref_idx_l1 < 0),
            BScenario::Uni1 => coded.clone().any(|d| d.ref_idx < 0 && d.ref_idx_l1 >= 0),
            BScenario::Bi => coded.clone().any(|d| d.ref_idx >= 0 && d.ref_idx_l1 >= 0),
        };
        assert!(
            hit,
            "{tag}: the scenario never reached its own prediction shape"
        );

        // The independent state.
        let geo = std::sync::Arc::new(Geometry::new(&sps, &pps));
        let mut info = PicInfo::new(geo);
        let mut frame = Frame::<u8>::new(64, 64, chroma, 8);
        frame.poc = 2;
        let mut scratch = McScratch::new();
        let no_backward_pred = [ref0.poc, ref1.poc].iter().all(|&p| p <= 2);
        assert!(
            !no_backward_pred,
            "a future anchor must make NoBackwardPredFlag false"
        );
        let refs = RefCtx::<u8> {
            pocs: [vec![ref0.poc], vec![ref1.poc]],
            long_term: [vec![false], vec![false]],
            col: None,
            cur_poc: 2,
            no_backward_pred,
            tmvp: false,
            max_merge_cand: MAX_MERGE_CAND,
            log2_par_mrg_level: 2,
            is_b: true,
            num_ref_idx: [1, 1],
            col_from_l0: true,
        };
        for (i, d) in decisions.iter().enumerate() {
            let (cx, cy) = (i % (64 / n), i / (64 / n));
            let (x0, y0) = (cx * n, cy * n);
            let ctb = info.ctb_of(x0, y0);
            info.ctb_slice_addr[ctb] = 0;
            info.ctb_slice[ctb] = 0;
            let pu = PuPos {
                x_cb: x0 as i32,
                y_cb: y0 as i32,
                n_cb: n as i32,
                x_pb: x0 as i32,
                y_pb: y0 as i32,
                w: n as i32,
                h: n as i32,
                part_idx: 0,
            };
            let w4 = info.w4;
            let (mv, ref_idx) = match d.kind {
                InterCuKind::Skip { merge_idx } | InterCuKind::Merge { merge_idx } => {
                    let cand = merge_candidate(&info, &frame, &refs, &pu, merge_idx as usize);
                    (cand.mv, cand.ref_idx)
                }
                InterCuKind::BAmvp { idc, mvd, mvp_flag } => {
                    let r: [i8; 2] = match idc {
                        0 => [0, -1],
                        1 => [-1, 0],
                        _ => [0, 0],
                    };
                    let mut mv = [Mv::ZERO; 2];
                    for list in 0..2usize {
                        if r[list] >= 0 {
                            let p = amvp(&info, &frame, &refs, &pu, list, 0, mvp_flag[list] as u32);
                            mv[list] = Mv::new(
                                p.x.wrapping_add(mvd[list].x),
                                p.y.wrapping_add(mvd[list].y),
                            );
                        }
                    }
                    (mv, r)
                }
                InterCuKind::Amvp { .. } => unreachable!("the B walk never produces the P shape"),
                InterCuKind::Parts => unreachable!(
                    "this replay codes whole units: the walk it drives leaves partitions off"
                ),
                InterCuKind::UseIntra => {
                    fill_motion(&mut frame.motion, frame.w4, x0, y0, n, n, MotionInfo::INTRA);
                    PicInfo::fill4(&mut info.pred_mode, w4, x0, y0, n, n, 1);
                    continue;
                }
            };
            assert_eq!(
                ref_idx,
                [d.ref_idx, d.ref_idx_l1],
                "cu {i}: replayed lists differ ({:?})",
                d.kind
            );
            assert_eq!(
                [mv[0], mv[1]],
                [d.mv, d.mv_l1],
                "cu {i}: signalling does not replay to the chosen vectors ({:?})",
                d.kind
            );

            let r0 = (ref_idx[0] >= 0).then_some((&ref0, mv[0]));
            let r1 = (ref_idx[1] >= 0).then_some((&ref1, mv[1]));
            predict_block(
                &kit.dsp,
                &mut scratch,
                &mut frame,
                x0,
                y0,
                n,
                n,
                r0,
                r1,
                weighting(ref_idx),
            );
            if d.rqt_root_cbf {
                let bd_shift = 20 - 8i32;
                let mut work = [0i16; 1024];
                if d.cbf_luma {
                    work[..n * n].copy_from_slice(&d.luma[..n * n]);
                    let log2 = d.log2_cu;
                    scale_coefficients(
                        &mut work,
                        log2,
                        ctx.qp,
                        8,
                        ScalingSource::Flat,
                        false,
                        n - 1,
                        n - 1,
                    );
                    (kit.dsp.idct[(log2 - 2) as usize])(&mut work, bd_shift, n - 1, n - 1);
                    let off = frame.y.offset(x0 as isize, y0 as isize);
                    (kit.dsp.add_residual)(&mut frame.y.data[off..], frame.y.stride, &work, n, 255);
                }
                if cat != 0 {
                    let qp_c = chroma_qp(cat, ctx.qp.clamp(0, 57));
                    let (sw, sh) = sub_wh(cat);
                    let (tbs, ntb, log2c) = chroma_tbs(cat, x0, y0, d.log2_cu);
                    let nc = 1usize << log2c;
                    let nc2 = nc * nc;
                    for comp in 0..2 {
                        for (t, &(ax, ay)) in tbs[..ntb].iter().enumerate() {
                            let cbf = if t == 0 {
                                d.cbf_chroma[comp]
                            } else {
                                d.cbf_chroma_bot[comp]
                            };
                            if !cbf {
                                continue;
                            }
                            work[..nc2].copy_from_slice(&d.chroma[comp][t * nc2..(t + 1) * nc2]);
                            scale_coefficients(
                                &mut work,
                                log2c,
                                qp_c,
                                8,
                                ScalingSource::Flat,
                                false,
                                nc - 1,
                                nc - 1,
                            );
                            (kit.dsp.idct[(log2c - 2) as usize])(
                                &mut work,
                                bd_shift,
                                nc - 1,
                                nc - 1,
                            );
                            let plane = if comp == 0 {
                                &mut frame.cb
                            } else {
                                &mut frame.cr
                            };
                            let off = plane.offset((ax / sw) as isize, (ay / sh) as isize);
                            (kit.dsp.add_residual)(
                                &mut plane.data[off..],
                                plane.stride,
                                &work,
                                nc,
                                255,
                            );
                        }
                    }
                }
            }
            let mut mi = MotionInfo {
                mv,
                ref_idx,
                ref_delta: [0; 2],
                flags: 0,
                pad: 0,
            };
            for list in 0..2usize {
                if ref_idx[list] >= 0 {
                    let poc = if list == 0 { ref0.poc } else { ref1.poc };
                    mi.ref_delta[list] = (2 - poc).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                }
            }
            fill_motion(&mut frame.motion, frame.w4, x0, y0, n, n, mi);
            PicInfo::fill4(&mut info.pred_mode, w4, x0, y0, n, n, 0);
        }
        assert_eq!(
            frame.y.data, pic.recon.y.data,
            "{tag}: luma reconstruction differs from the replay"
        );
        assert_eq!(
            frame.cb.data, pic.recon.cb.data,
            "{tag}: cb reconstruction differs from the replay"
        );
        assert_eq!(
            frame.cr.data, pic.recon.cr.data,
            "{tag}: cr reconstruction differs from the replay"
        );
        assert_eq!(
            frame.motion, pic.recon.motion,
            "{tag}: motion grids diverged"
        );
    }

    /// The anchor: every decision, replayed through the decoder's own
    /// candidate derivation over an *independently maintained* state —
    /// fresh `PicInfo`, fresh motion grid, fresh reconstruction — the way
    /// a writer plus a decoder would consume it. This proves the state
    /// maintenance (motion fills, availability marks), which is the half
    /// of the signalling contract that direct reuse of `merge_candidate` /
    /// `amvp` does not already make unbreakable; the derivation itself is
    /// held by the decoder's conformance suites.
    ///
    /// Run once per chroma format, because the reconstruction the replay
    /// rebuilds is where each format's chroma differs: the vector
    /// `predict_block` derives, the transform blocks the residual lands
    /// in, and the QP mapping that scales it. A 4:2:0-only replay would
    /// pass with every other format's chroma wrong.
    #[test]
    fn every_decision_replays_through_an_independent_decoder_state() {
        for chroma in [
            ChromaFormat::Monochrome,
            ChromaFormat::Yuv420,
            ChromaFormat::Yuv422,
            ChromaFormat::Yuv444,
        ] {
            replay_one_format(chroma);
        }
    }

    fn replay_one_format(chroma: ChromaFormat) {
        let kit = Kit::new();
        let ctx = kit.ctx(30);
        let (sps, pps) = parsed_sets_fmt(64, 64, chroma);
        let cat = sps.chroma_array_type();
        let refp = reference_fmt(64, 64, 17, chroma);
        // Mixed content: a translation with damage in some regions, so
        // skip, merge-with-residual and AMVP all appear.
        let (mut sy, mut scb, mut scr) = translated(&refp, 3, 1);
        let mut s = 5u64;
        for yy in 0..64usize {
            for xx in 0..64usize {
                if (xx / 16 + yy / 16) % 3 == 0 {
                    let d = (lcg(&mut s) % 48) as i32 - 24;
                    let v = &mut sy[yy * 64 + xx];
                    *v = (*v as i32 + d).clamp(0, 255) as u8;
                }
            }
        }
        // Chroma damage as well, in a different stripe, because at 4:4:4
        // the chroma translation is exact (SubWidthC and SubHeightC are 1,
        // so the fixture's integer shift *is* the true chroma vector) and
        // an undamaged source leaves no chroma residual at all — the
        // replay would then compare two copies of a plain prediction and
        // prove nothing about the chroma transform path. 4:2:0 and 4:2:2
        // get residual for free from their fractional chroma vectors; this
        // makes every format carry some.
        let (cw, ch) = (refp.cb.width, refp.cb.height);
        for yy in 0..ch {
            for xx in 0..cw {
                if (xx / 8 + yy / 8) % 3 == 1 {
                    let d = (lcg(&mut s) % 40) as i32 - 20;
                    let v = &mut scb[yy * cw + xx];
                    *v = (*v as i32 + d).clamp(0, 255) as u8;
                    let d = (lcg(&mut s) % 40) as i32 - 20;
                    let v = &mut scr[yy * cw + xx];
                    *v = (*v as i32 + d).clamp(0, 255) as u8;
                }
            }
        }
        let (pic, decisions) = code_picture(&ctx, &sps, &pps, &refp, &sy, &scb, &scr);
        assert_invariants(&decisions, cat);
        let kinds: Vec<_> = decisions
            .iter()
            .map(|d| std::mem::discriminant(&d.kind))
            .collect();
        assert!(
            kinds.iter().collect::<std::collections::HashSet<_>>().len() >= 2,
            "one-note content: the replay would prove less ({chroma:?})"
        );
        // A format with chroma must actually exercise it, or the replay's
        // chroma comparison below proves nothing about that format.
        if cat != 0 {
            assert!(
                decisions.iter().any(|d| d.cbf_chroma[0]
                    || d.cbf_chroma[1]
                    || d.cbf_chroma_bot[0]
                    || d.cbf_chroma_bot[1]),
                "{chroma:?}: no CU carried a chroma residual"
            );
        }
        if cat == 2 {
            assert!(
                decisions
                    .iter()
                    .any(|d| d.cbf_chroma_bot[0] || d.cbf_chroma_bot[1]),
                "4:2:2: the stacked pair's bottom square never carried anything"
            );
        }

        // The independent state.
        let geo = std::sync::Arc::new(Geometry::new(&sps, &pps));
        let mut info = PicInfo::new(geo);
        let mut frame = Frame::<u8>::new(64, 64, chroma, 8);
        frame.poc = 1;
        let mut scratch = McScratch::new();
        let refs = RefCtx::<u8> {
            pocs: [vec![refp.poc], Vec::new()],
            long_term: [vec![false], Vec::new()],
            col: None,
            cur_poc: 1,
            no_backward_pred: true,
            tmvp: false,
            max_merge_cand: MAX_MERGE_CAND,
            log2_par_mrg_level: 2,
            is_b: false,
            num_ref_idx: [1, 0],
            col_from_l0: true,
        };
        let n = 1usize << sps.log2_ctb_size;
        for (i, d) in decisions.iter().enumerate() {
            let (cx, cy) = (i % (64 / n), i / (64 / n));
            let (x0, y0) = (cx * n, cy * n);
            let ctb = info.ctb_of(x0, y0);
            info.ctb_slice_addr[ctb] = 0;
            info.ctb_slice[ctb] = 0;
            let pu = PuPos {
                x_cb: x0 as i32,
                y_cb: y0 as i32,
                n_cb: n as i32,
                x_pb: x0 as i32,
                y_pb: y0 as i32,
                w: n as i32,
                h: n as i32,
                part_idx: 0,
            };
            let w4 = info.w4;
            let mv = match d.kind {
                InterCuKind::Skip { merge_idx } | InterCuKind::Merge { merge_idx } => {
                    let cand = merge_candidate(&info, &frame, &refs, &pu, merge_idx as usize);
                    assert_eq!(
                        cand.ref_idx,
                        [0, -1],
                        "cu {i}: replayed merge candidate references differently"
                    );
                    cand.mv[0]
                }
                InterCuKind::Amvp { mvp_flag, mvd } => {
                    let mvp = amvp(&info, &frame, &refs, &pu, 0, 0, mvp_flag as u32);
                    Mv::new(mvp.x.wrapping_add(mvd.x), mvp.y.wrapping_add(mvd.y))
                }
                InterCuKind::BAmvp { .. } => {
                    unreachable!("this replay drives the P walk, which never produces a B shape")
                }
                InterCuKind::Parts => unreachable!(
                    "this replay codes whole units: the walk it drives leaves partitions off"
                ),
                InterCuKind::UseIntra => {
                    fill_motion(&mut frame.motion, frame.w4, x0, y0, n, n, MotionInfo::INTRA);
                    PicInfo::fill4(&mut info.pred_mode, w4, x0, y0, n, n, 1);
                    continue;
                }
            };
            assert_eq!(
                mv, d.mv,
                "cu {i}: the signalling does not replay to the chosen vector ({:?})",
                d.kind
            );

            // Reconstruct as a decoder would: the prediction, then the
            // carried coefficients through the inverse path.
            predict_block(
                &kit.dsp,
                &mut scratch,
                &mut frame,
                x0,
                y0,
                n,
                n,
                Some((&refp, mv)),
                None,
                [Weighting::Default; 3],
            );
            if d.rqt_root_cbf {
                let bd_shift = 20 - 8i32;
                let mut work = [0i16; 1024];
                if d.cbf_luma {
                    work[..n * n].copy_from_slice(&d.luma[..n * n]);
                    let log2 = d.log2_cu;
                    scale_coefficients(
                        &mut work,
                        log2,
                        ctx.qp,
                        8,
                        ScalingSource::Flat,
                        false,
                        n - 1,
                        n - 1,
                    );
                    (kit.dsp.idct[(log2 - 2) as usize])(&mut work, bd_shift, n - 1, n - 1);
                    let off = frame.y.offset(x0 as isize, y0 as isize);
                    (kit.dsp.add_residual)(&mut frame.y.data[off..], frame.y.stride, &work, n, 255);
                }
                if cat != 0 {
                    let qp_c = chroma_qp(cat, ctx.qp.clamp(0, 57));
                    let (sw, sh) = sub_wh(cat);
                    // Placed by the reader's own derivation, not by this
                    // test's arithmetic: `chroma_tbs` is `transform_unit`'s
                    // `here` plus its stacked-pair loop.
                    let (tbs, ntb, log2c) = chroma_tbs(cat, x0, y0, d.log2_cu);
                    let nc = 1usize << log2c;
                    let nc2 = nc * nc;
                    for comp in 0..2 {
                        for (t, &(ax, ay)) in tbs[..ntb].iter().enumerate() {
                            let cbf = if t == 0 {
                                d.cbf_chroma[comp]
                            } else {
                                d.cbf_chroma_bot[comp]
                            };
                            if !cbf {
                                continue;
                            }
                            work[..nc2].copy_from_slice(&d.chroma[comp][t * nc2..(t + 1) * nc2]);
                            scale_coefficients(
                                &mut work,
                                log2c,
                                qp_c,
                                8,
                                ScalingSource::Flat,
                                false,
                                nc - 1,
                                nc - 1,
                            );
                            (kit.dsp.idct[(log2c - 2) as usize])(
                                &mut work,
                                bd_shift,
                                nc - 1,
                                nc - 1,
                            );
                            let plane = if comp == 0 {
                                &mut frame.cb
                            } else {
                                &mut frame.cr
                            };
                            let off = plane.offset((ax / sw) as isize, (ay / sh) as isize);
                            (kit.dsp.add_residual)(
                                &mut plane.data[off..],
                                plane.stride,
                                &work,
                                nc,
                                255,
                            );
                        }
                    }
                }
            }

            // The decoder's own motion store, on the replay's state.
            let mut mi = MotionInfo {
                mv: [mv, Mv::ZERO],
                ref_delta: [0; 2],
                ref_idx: [0, -1],
                flags: 0,
                pad: 0,
            };
            mi.ref_delta[0] = (1 - refp.poc).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            fill_motion(&mut frame.motion, frame.w4, x0, y0, n, n, mi);
            PicInfo::fill4(&mut info.pred_mode, w4, x0, y0, n, n, 0);
        }
        // The replayed reconstruction is the encoder's, byte for byte.
        assert_eq!(
            frame.y.data, pic.recon.y.data,
            "{chroma:?}: luma reconstruction differs from the replay"
        );
        assert_eq!(
            frame.cb.data, pic.recon.cb.data,
            "{chroma:?}: cb reconstruction differs from the replay"
        );
        assert_eq!(
            frame.cr.data, pic.recon.cr.data,
            "{chroma:?}: cr reconstruction differs from the replay"
        );
        // And the two sides' motion state agrees, which is what the next
        // picture would predict TMVP from if the SPS ever enables it.
        assert_eq!(frame.motion, pic.recon.motion, "motion grids diverged");
    }
}
