//! The H.265 encoder's speed tiers: what each [`Preset`] sets.
//!
//! A tier is two things. The *structure* it implies — B pictures, the
//! pyramid, references, SAO, inter partitions — lives in ordinary
//! [`Config`] fields, which [`apply_preset`] sets to the tier's defaults
//! so that a caller can still override any of them afterwards. The
//! *tuning* — quantiser offsets per picture kind, search effort, early
//! exits — has no field of its own and is read off `Tune::of`, which
//! is all-neutral when no tier was asked for: a configuration without a
//! preset codes exactly as it did before tiers existed.

use super::{Config, InterParts, Preset};

/// Set the structural fields of `cfg` to what tier `p` codes with, and
/// record the tier. Call before applying explicit overrides.
pub fn apply_preset(cfg: &mut Config, p: Preset) {
    cfg.preset = p;
    cfg.bframes = 3;
    cfg.pyramid = true;
    cfg.sao = true;
    // H.264's tools, which the shared `--preset` sets: not H.265 syntax.
    cfg.transform_8x8 = false;
    cfg.subparts = false;
    // H.265's lookahead propagation (`h265_cutree`) sets only the intra
    // pictures' quantiser offset under a tier (`Tune::tree_frame_only`),
    // from sixteen pictures of lookahead.
    cfg.tree_lookahead = 16;
    match p {
        Preset::Veryfast => {
            cfg.max_refs = 1;
            cfg.inter_parts = InterParts::None;
        }
        Preset::Medium => {
            cfg.max_refs = 2;
            cfg.inter_parts = InterParts::None;
        }
        // Symmetric partitions at slow, searched from the whole unit's
        // motion (`Tune::part_seeded`, round 5): see `Tune::of`. Measured
        // first (2026-11-15) searched from scratch: +6.38% against +6.28%
        // BD-rate without them, at 1.9x the CPU.
        Preset::Slow => {
            cfg.max_refs = 4;
            cfg.inter_parts = InterParts::Symmetric;
        }
    }
}

/// The tuning a tier implies beyond the structural fields.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Tune {
    /// Quantiser offset of an intra picture against the configured QP,
    /// which is a P picture's; under `tree_frame_only` the propagation's
    /// offset replaces it.
    pub i_qp_offset: i32,
    /// Quantiser offset of a B picture by its layer
    /// ([`super::gop::Coded::layer`]): index 0 is layer 1, the top of a
    /// pyramid or every B of a flat GOP; deeper layers take the last
    /// entry.
    pub b_qp_offset: [i32; 3],
    /// How many times the full-sample motion search widens to square
    /// rings at 4, 8 and 16 samples once its descent settles. Measured
    /// 2026-11-15 (x265 parity, `--preset medium`): one widening -0.71%,
    /// none -0.49%, three -0.73%, at 1.06x / 1.00x / 1.14x the CPU.
    pub search_rings: u32,
    /// Intra in inter pictures by coding it, under this cost gate
    /// (`h265_me::InterPicture::rd_intra`), or by the flat-block proxy.
    pub rd_intra: Option<f32>,
    /// The recursion skip (`h265_me::InterPicture::rskip`).
    pub rskip: Option<f32>,
    /// How deep an intra unit inside an inter picture may split its
    /// transform. Measured 2026-11-15 (x265 parity, `--preset medium`, CPU
    /// by interleaved one-thread runs): 0 against 1, -4.19% against -4.28%
    /// BD-rate at 0.92x the CPU.
    pub intra_split_depth: u32,
    /// Sign data hiding (the PPS flag, `h265_rdoq::hide_signs`).
    pub sign_hiding: bool,
    /// Rate-distortion optimised quantisation (`h265_rdoq::quantise`).
    pub rdoq: bool,
    /// RDOQ for the chroma of inter units too
    /// (`h265_me::InterPicture::rdoq_chroma`); intra units quantise
    /// chroma by it wherever `rdoq` is on.
    pub rdoq_inter_chroma: bool,
    /// RDOQ in pictures nothing references as well.
    pub rdoq_nonref: bool,
    /// SAO (under `Config::sao`) in pictures nothing references as well;
    /// otherwise their slices carry the SAO flags clear.
    pub sao_nonref: bool,
    /// How deep an intra picture's units may split their transform.
    pub i_split_depth: u32,

    /// `h265_me::InterPicture::drop_trial`.
    pub drop_trial: bool,
    /// `h265_me::InterPicture::nxn_trial`.
    pub nxn_trial: f64,
    /// `h265_me::InterPicture::quarter_cross`.
    pub quarter_cross: bool,
    /// `h265_me::InterPicture::intra_satd_gate`.
    pub intra_satd_gate: Option<f32>,
    /// `h265_intra::IntraCtx::fast_intra`.
    pub fast_intra: bool,
    /// A picture nothing references decides with the cheapest settings:
    /// every skip ends the recursion, intra only where its SATD estimate
    /// beats the inter search, no `PART_NxN` trial.
    pub nonref_fast: bool,
    /// `h265_me::InterPicture::limit_refs`.
    pub limit_refs: bool,
    /// Lookahead quantiser propagation strength (`h265_cutree`), applied
    /// over `Config::tree_lookahead` pictures when that is nonzero (the
    /// shared `--tree`). Per unit it lost when first measured (2026-11-15,
    /// medium: -3.31% BD-rate to +2.58% at 2.0, to -1.57% at 1.0), where
    /// x265 loses 14-22% on the 720p clips without its own; the tiers use
    /// it for an intra picture's offset alone (`tree_frame_only`).
    pub cutree: f32,
    /// Decisions price their bins at the contexts the picture has
    /// reached (each CTU decided is run through the writer), rather than
    /// at the slice's initial ones.
    pub live_contexts: bool,
    /// Temporal motion vector prediction (the SPS flag; every inter slice
    /// then names its collocated picture).
    pub tmvp: bool,
    /// `h265_me::InterPicture::tu_split`.
    pub inter_tu_split: bool,
    /// The largest inter CU (log2) the split is tried in.
    pub inter_tu_split_max: u32,
    /// `h265_me::InterPicture::bi_refine`.
    pub bi_refine: u32,
    /// `h265_me::InterPicture::bi_refine_gate`.
    pub bi_refine_gate: f32,
    /// `h265_me::InterPicture::tu_split_nz`.
    pub tu_split_nz: usize,
    /// `h265_me::InterPicture::merge_tried`.
    pub merge_tried: usize,
    /// `h265_me::InterPicture::split_gate`.
    pub split_gate: Option<f32>,
    /// `h265_me::InterPicture::split_gate_above`.
    pub split_gate_above: u32,
    /// The Lagrangian multiplier of an intra picture, a P picture, the
    /// reference B and the other B pictures
    /// (`h265_intra::set_lambda_scale`).
    pub lambda: [f32; 4],
    /// Under `Config::tree_lookahead`, the propagation sets only an intra
    /// picture's quantiser offset, from its mean over the picture,
    /// replacing `i_qp_offset`; no unit carries a quantiser delta.
    pub tree_frame_only: bool,
    /// The most an intra picture's propagated offset may lower it.
    pub i_adapt_min: i32,
    /// The intra offset is `i_adapt_k * (mean - i_adapt_m0)`, rounded.
    pub i_adapt_k: f32,
    pub i_adapt_m0: f32,
    /// References per list in a B slice (`h265_me::MAX_B_REFS` at most).
    pub b_refs: usize,
    /// `h265_me::InterPicture::b_limit_refs`.
    pub b_limit_refs: bool,
    /// Where RDOQ starts its last-position search, in steps
    /// (`h265_rdoq::set_last_from`); 0.5 weighs every coefficient that
    /// rounds to a level.
    pub rdoq_last: f32,
    /// How many SAO edge classes each CTB weighs (`h265_sao::set_edge_classes`).
    pub sao_edge_classes: u8,
    /// `h265_me::InterPicture::part_seeded`.
    pub part_seeded: bool,
    /// `h265_me::InterPicture::parts_after_merge`.
    pub parts_after_merge: bool,
    /// `h265_me::InterPicture::parts_rd_margin`.
    pub parts_rd_margin: f32,
}

impl Tune {
    /// Nothing moved: what a configuration without a tier codes with.
    pub const NEUTRAL: Tune = Tune {
        i_qp_offset: 0,
        b_qp_offset: [0; 3],
        search_rings: 0,
        rd_intra: Some(0.5),
        rskip: None,
        intra_split_depth: 1,
        sign_hiding: false,
        rdoq: false,
        rdoq_inter_chroma: true,
        rdoq_nonref: true,
        sao_nonref: true,
        i_split_depth: 1,
        drop_trial: true,
        nxn_trial: 2.0,
        quarter_cross: false,
        intra_satd_gate: None,
        fast_intra: false,
        nonref_fast: false,
        limit_refs: false,
        cutree: 2.0,
        live_contexts: false,
        tmvp: false,
        inter_tu_split: false,
        inter_tu_split_max: 5,
        bi_refine: 0,
        bi_refine_gate: f32::INFINITY,
        tu_split_nz: 0,
        merge_tried: 5,
        split_gate: None,
        split_gate_above: 3,
        lambda: [1.0; 4],
        tree_frame_only: false,
        i_adapt_min: -6,
        i_adapt_k: 4.0,
        i_adapt_m0: -1.4,
        b_refs: 1,
        b_limit_refs: false,
        rdoq_last: 0.5,
        sao_edge_classes: 4,
        part_seeded: false,
        parts_after_merge: true,
        parts_rd_margin: 1.0,
    };

    /// The tuning for `cfg`'s tier, or [`Tune::NEUTRAL`] without one.
    pub fn of(cfg: &Config) -> Tune {
        if cfg.legacy_decisions {
            return Tune::NEUTRAL;
        }
        let p = cfg.preset;
        // Offsets measured 2026-11-15 (x265 parity, 30 frames of the five
        // clips, `--bframes 3 --refs 4 --sao --b-pyramid`, BD-rate PSNR-Y
        // mean against x265 medium): I/B1/B2 at -3/+2/+4 +1.85%, -4/+2/+5
        // +1.83%, -3/+3/+5 +1.96%; a flat GOP at its best (I -3, B +3)
        // +12.0%, a 5- or 7-picture pyramid at -3/+2/+3/+4 +6.3% / +4.7%.
        let base = Tune {
            i_qp_offset: -3,
            b_qp_offset: [2, 4, 5],
            // Measured 2026-11-15 (x265 parity against medium, `--preset
            // medium`): sign hiding -4.19% -> -6.06% BD-rate; RDOQ alone
            // -7.47% (ducks +10.7 -> -3.0, park_joy +6.0 -> +1.0, the CIF
            // clips 0.3-1.0 worse) and both -8.25%, at 1.3x the CPU for
            // RDOQ by interleaved one-thread runs. See `rdoq` below for
            // where it stands now.
            sign_hiding: true,
            // Measured 2026-11-15 (x265 parity, `--preset medium`):
            // pricing at live contexts -3.35% -> -4.17% BD-rate, and
            // temporal MVP on top -5.31% (veryfast -5.09% -> -6.97%, slow
            // +5.96% -> +3.93% with RDOQ kept on the initial contexts).
            live_contexts: true,
            tmvp: true,
            // Measured 2026-11-15 (x265 parity): splitting an inter CU's
            // transform once where that is cheaper, 8x8 CUs included,
            // took medium from -5.31% to -8.18% BD-rate (park_joy +3.31%
            // -> -0.58%), veryfast -6.74% -> -9.98%, slow +4.03% ->
            // +2.48% (mobile +10.25% -> +5.96%), for a fifth more CPU.
            // Without the 8x8 CUs medium gained only -0.74%.
            inter_tu_split: true,
            // 8x8 CUs only: most of the gain at a fraction of the CPU
            // (medium -7.53% at 1.05x against -8.18% at 1.19x with 16x16
            // and 32x32 too; slow +2.53% at 1.10x against +2.48% at 1.27x;
            // veryfast -9.11% at 1.22x against -9.98% at 1.41x).
            inter_tu_split_max: 3,
            // Every tier quantises by RDOQ (round 3, 2026-11-15, x265
            // parity). RDOQ on the chroma of inter units lost at every
            // tier in round 3 (medium -8.78% BD-rate with it, -10.38%
            // without) because the quantiser could not leave a block
            // empty: its last position was forced to a level. Round 5
            // (2026-11-15) weighs every block against coding nothing
            // (`h265_rdoq::Block::cbf_ctx`) and prices the coefficient
            // group flags in the last-position search; with both, inter
            // chroma by RDOQ wins (slow -2.56% -> -3.28%, medium -13.12%
            // -> -14.25%, veryfast -12.54% -> -13.76%, PSNR-Y). At
            // medium and veryfast the pictures nothing references skip
            // it: -10.28% against -10.38% at medium for a twentieth less
            // CPU; veryfast -9.11% -> -11.76% (ducks +0.50% -> -11.03%)
            // at 1.15x the CPU.
            rdoq: true,
            rdoq_inter_chroma: true,
            rdoq_nonref: false,
            // No SAO in the pictures nothing references (round 3,
            // 2026-11-15, x265 parity): veryfast -11.76% -> -12.06%
            // BD-rate at 0.92x the CPU, medium -10.28% -> -10.55% at
            // 0.94x, slow +0.79% -> +0.58% at 0.95x.
            sao_nonref: false,
            // An intra picture's quantiser offset from the propagation's
            // mean over it, in place of a fixed -3 (round 4, 2026-11-15,
            // x265 parity): the 720p clips, whose content later pictures
            // inherit little of, want their intra pictures at the P
            // pictures' quantiser (ducks +7.96% at -3, +5.78% at 0); the
            // CIF clips want -3 or below (akiyo -13.81% at -3, -16.42% at
            // -5). The propagation's means at strength 1 over sixteen
            // pictures — foreman -2.1, mobile -2.0, akiyo -3.8, park_joy
            // -0.8, ducks -1.4 — order them so: slow +0.58% -> -0.59%,
            // medium -10.55% -> -11.45%, veryfast -11.88% -> -12.52%, at
            // the same CPU.
            tree_frame_only: true,
            cutree: 1.0,
            i_adapt_k: 4.5,
            i_adapt_m0: -1.45,
            // RDOQ's last position searched from the last coefficient of
            // 0.7 of a step (round 5, 2026-11-15, x265 parity, on the
            // round's other changes): slow -5.85% -> -6.00% BD-rate at
            // 0.98x the CPU, medium -14.40% -> -14.45% at 0.94x, veryfast
            // -13.91% -> -14.09% at 0.98x; at 0.85 slow -5.10%, medium
            // -14.25%, veryfast -13.91% at 0.96x.
            rdoq_last: 0.7,
            ..Tune::NEUTRAL
        };
        match p {
            // Intra units in inter pictures without a transform split, and
            // tried only where their SATD estimate comes within 1.5x of
            // the inter search: measured 2026-11-15, +3.93% -> +4.03%
            // BD-rate at 0.78x the CPU.
            // Round 4 (2026-11-15, x265 parity): one round of iterative
            // bi-prediction refinement, +0.58% -> +0.01% BD-rate on its
            // own at 0.90x x265's CPU against 0.82x; every Lagrangian at
            // 0.7 of the conventional one (alone on round 3: 0.8 -0.36%,
            // 0.65 -0.50%, 0.5 +0.30%, the CIF clips losing below 0.8 and
            // the 720p clips wanting it lower still). With the intra
            // offset above, both: -0.59% -> -2.30% at 0.93x (-2.09% at
            // a Lagrangian of 0.8).
            // Round 5 (2026-11-15, x265 parity): two references per list
            // in B slices, the further one searched below the top of the
            // tree only where the enclosing unit chose it: -4.75% ->
            // -5.51% BD-rate (mobile -1.08% -> -5.02%) at 1.06x the CPU;
            // searched everywhere -5.75% at 1.17x; four per list, limited,
            // -5.71% at 1.09x. Medium gained -0.4% for 1.05-1.09x.
            // Round 5 (2026-11-15, x265 parity): symmetric partitions
            // (`Config::inter_parts`, set by `apply_preset`) searched from
            // the 2Nx2N unit's motion (`part_seeded`), tried only where the
            // 2Nx2N unit is not a merge, the loser coded too only within
            // 0.1 of the winner; with a skip cheaper than 0.5 Lagrangians a
            // sample ending the recursion (`rskip`): -6.00% -> -7.06%
            // BD-rate (foreman -5.9% -> -8.1%, park_joy +0.4% -> -0.8%)
            // at 0.93x x265's CPU. Partitions searched from scratch were
            // -8.23% at 2.1x; seeded, -7.73% at 1.65x; seeded behind the
            // merge gate, -6.99% at 1.16x; and rskip at 0.5 bought the
            // rest of the CPU back for -0.3%.
            Preset::Slow => Tune {
                part_seeded: true,
                parts_after_merge: false,
                parts_rd_margin: 0.1,
                rskip: Some(0.5),
                b_refs: 2,
                b_limit_refs: true,
                lambda: [0.7; 4],
                bi_refine: 1,
                // Refinement only where the unrefined pair comes within
                // 1.1x of the better one-list shape: -2.30% -> -2.31%
                // BD-rate at 0.95x the CPU (0.9: -2.03% at 0.90x).
                bi_refine_gate: 1.1,
                rdoq_nonref: true,
                search_rings: 1,
                intra_split_depth: 0,
                intra_satd_gate: Some(1.5),
                ..base
            },
            // The 8x8 transform split tried only where the whole TU kept
            // four luma levels or more (round 4, 2026-11-15, x265 parity):
            // -11.45% -> -11.33% BD-rate at 0.98x the CPU.
            Preset::Medium => Tune {
                tu_split_nz: 4,
                rskip: Some(0.5),
                intra_split_depth: 0,
                intra_satd_gate: Some(1.0),
                fast_intra: true,
                nxn_trial: 0.0,
                nonref_fast: true,
                ..base
            },
            // Veryfast (measured 2026-11-15 against x265 veryfast): with
            // medium's settings -9.17% BD-rate at 1.47x the CPU; with every
            // skip ending the recursion, intra by the flat-block proxy and
            // one reference -5.09% at 1.27x.
            // Intra pictures without a transform split (round 3): -12.06%
            // -> -11.88% BD-rate at 0.97x the CPU.
            // Round 4 (2026-11-15, x265 parity): a unit above 8x8 whose
            // 2Nx2N search costs more than 2.5 SATD Lagrangians a sample
            // goes straight to its children (`split_gate`), the 8x8
            // transform split only where the whole TU kept four levels or
            // more, and three merge candidates scored of five: -12.52% ->
            // -10.49% BD-rate at 0.89x the CPU (interleaved one-thread
            // runs; foreman and mobile 0.93x, the 720p clips 0.88x).
            // Measured on the way: the gate at 3 with the level count
            // -11.32% at 0.92x; the gate alone at 1.5 -10.02% (ducks
            // -3.87%), at 3 -11.38%;
            // the gate on 32x32 and 64x64 units only -12.02%, at 64x64
            // only never firing; no unit coded whole at 64x64 -8.93% at
            // no saving; SAO off in reference pictures too -10.29% for
            // 3.5%; quarter-sample refinement on a cross -10.19% (mobile
            // -6.97% -> -4.11%) for 2%; the merge candidates no longer
            // seeding a unit with a parent's vector, no saving.
            // Round 5 (2026-11-15, x265 parity, timed against x265 in
            // concurrent one-thread runs): RDOQ's last position from 0.85
            // of a step, two SAO edge classes of four and the early split
            // at 2.0: -14.09% -> -13.38% BD-rate at about 0.97x x265's CPU
            // from 1.05x. Measured on the way: no RDOQ in inter units at
            // all, -9.48% (ducks +0.59%) at 0.94x; no inter transform split
            // -12.04% (mobile -5.4%) at 0.95x; inter chroma by the plain
            // quantiser 1.7% for 1.2%; fewer merge candidates, nothing.
            Preset::Veryfast => Tune {
                rdoq_last: 0.85,
                sao_edge_classes: 2,
                split_gate: Some(2.0),
                tu_split_nz: 4,
                merge_tried: 3,
                i_split_depth: 0,
                rskip: Some(f32::MAX),
                rd_intra: None,
                intra_split_depth: 0,
                fast_intra: true,
                nxn_trial: 0.0,
                nonref_fast: true,
                ..base
            },
        }
    }

    /// The Lagrangian multiplier of picture `c` (`Tune::lambda`).
    pub fn lambda_of(&self, c: &super::gop::Coded) -> f32 {
        use super::gop::Kind;
        self.lambda[match c.kind {
            Kind::Idr | Kind::I => 0,
            Kind::P => 1,
            Kind::B if c.reference => 2,
            Kind::B => 3,
        }]
    }

    /// The quantiser offset of a picture of `kind` at B layer `layer`.
    ///
    /// The offsets belong to the tier's GOP structure: they apply only under
    /// a B pyramid (`Config::pyramid`, which `--preset` sets), so a stream
    /// asked for at a constant quantiser without one keeps it constant.
    pub fn qp_offset(&self, cfg: &Config, c: &super::gop::Coded) -> i32 {
        use super::gop::Kind;
        if !cfg.pyramid {
            return 0;
        }
        match c.kind {
            Kind::Idr | Kind::I => self.i_qp_offset,
            Kind::P => 0,
            // Under the scheduler's pyramid the middle B picture is the one
            // reference: layer 1 above the others' layer 2.
            Kind::B => self.b_qp_offset[if c.reference { 0 } else { 1 }],
        }
    }
}
