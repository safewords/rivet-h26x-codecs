//! The H.265 encoder's speed tiers: what each [`Preset`] sets.
//!
//! A tier is two things. The *structure* it implies — B pictures, the
//! pyramid, references, SAO, inter partitions — lives in ordinary
//! [`Config`] fields, which [`apply_preset`] sets to the tier's defaults
//! so that a caller can still override any of them afterwards. The
//! *tuning* — quantiser offsets per picture kind, search effort, early
//! exits — has no field of its own and is read off [`Tune::of`], which
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
        // Symmetric partitions measured at slow (2026-11-15): +6.38% against
        // +6.28% BD-rate without them, at 1.9x the CPU. Off until they
        // earn it.
        Preset::Slow => {
            cfg.max_refs = 4;
            cfg.inter_parts = InterParts::None;
        }
    }
}

/// The tuning a tier implies beyond the structural fields.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Tune {
    /// Quantiser offset of an intra picture against the configured QP,
    /// which is a P picture's.
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
    /// shared `--tree`). No tier sets the window for H.265 yet: as first
    /// measured (2026-11-15, medium) propagation at 2.0 took BD-rate from
    /// -3.31% to +2.58% and at 1.0 to -1.57%, where x265 loses 14-22% on
    /// the 720p clips without its own.
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
        lambda: [1.0; 4],
        tree_frame_only: false,
        i_adapt_min: -6,
        i_adapt_k: 4.0,
        i_adapt_m0: -1.4,
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
            // parity): the luma of inter units, and intra units whole.
            // RDOQ on the chroma of inter units lost at every tier (medium
            // -8.78% BD-rate with it, -10.38% without; slow +2.52% /
            // +0.79%), and a chroma Lagrangian scaled by the chroma
            // quantiser's step did not recover it (-8.44% / +2.70%). At
            // medium and veryfast the pictures nothing references skip
            // it: -10.28% against -10.38% at medium for a twentieth less
            // CPU; veryfast -9.11% -> -11.76% (ducks +0.50% -> -11.03%)
            // at 1.15x the CPU.
            rdoq: true,
            rdoq_inter_chroma: false,
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
            Preset::Slow => Tune {
                lambda: [0.7; 4],
                bi_refine: 1,
                rdoq_nonref: true,
                search_rings: 1,
                intra_split_depth: 0,
                intra_satd_gate: Some(1.5),
                ..base
            },
            Preset::Medium => Tune {
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
            Preset::Veryfast => Tune {
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
