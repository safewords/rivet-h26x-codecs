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
    cfg.preset = Some(p);
    cfg.bframes = 3;
    cfg.b_pyramid = true;
    cfg.sao = true;
    match p {
        Preset::Veryfast => {
            cfg.max_refs = 2;
            cfg.inter_parts = InterParts::None;
        }
        Preset::Medium => {
            cfg.max_refs = 3;
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
    };

    /// The tuning for `cfg`'s tier, or [`Tune::NEUTRAL`] without one.
    pub fn of(cfg: &Config) -> Tune {
        let Some(p) = cfg.preset else {
            return Tune::NEUTRAL;
        };
        // Offsets measured 2026-11-15 (x265 parity, 30 frames of the five
        // clips, `--bframes 3 --refs 4 --sao --b-pyramid`, BD-rate PSNR-Y
        // mean against x265 medium): I/B1/B2 at -3/+2/+4 +1.85%, -4/+2/+5
        // +1.83%, -3/+3/+5 +1.96%; a flat GOP at its best (I -3, B +3)
        // +12.0%, a 5- or 7-picture pyramid at -3/+2/+3/+4 +6.3% / +4.7%.
        Tune {
            i_qp_offset: -3,
            b_qp_offset: [2, 4, 5],
            rskip: Some(0.25),
            intra_split_depth: 0,
            // Measured 2026-11-15 (x265 parity against medium, `--preset
            // medium`): sign hiding -4.19% -> -6.06% BD-rate; RDOQ alone
            // -7.47% (ducks +10.7 -> -3.0, park_joy +6.0 -> +1.0, the CIF
            // clips 0.3-1.0 worse) and both -8.25%, at 1.3x the CPU for
            // RDOQ by interleaved one-thread runs: the slow tier's tool.
            sign_hiding: true,
            rdoq: p == Preset::Slow,
            ..Tune::NEUTRAL
        }
    }

    /// The quantiser offset of a picture of `kind` at B layer `layer`.
    pub fn qp_offset(&self, kind: super::gop::Kind, layer: u8) -> i32 {
        use super::gop::Kind;
        match kind {
            Kind::Idr | Kind::I => self.i_qp_offset,
            Kind::P => 0,
            Kind::B => self.b_qp_offset[usize::from(layer.max(1) - 1).min(2)],
        }
    }
}
