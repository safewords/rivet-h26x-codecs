//! CABAC context variables for HEVC: the layout (offsets per syntax
//! element, matching the generated `CABAC_INIT` table) and initialisation
//! (9.3.2.2), plus the screen content coding contexts and the palette
//! predictor, which are stored and synchronised with them (9.3.2.4 /
//! 9.3.2.5).

use crate::cabac::{Ctx, init_ctx_hevc};

pub use super::tables_gen::*;

/// `palette_mode_flag` (Table 9-38): one context.
pub const PALETTE_MODE_FLAG_OFFSET: usize = NUM_CONTEXTS;
/// `tu_residual_act_flag` (Table 9-39): one context.
pub const TU_RESIDUAL_ACT_FLAG_OFFSET: usize = PALETTE_MODE_FLAG_OFFSET + 1;
/// `palette_run_prefix` (Table 9-40): eight contexts (9.3.4.2.8).
pub const PALETTE_RUN_PREFIX_OFFSET: usize = TU_RESIDUAL_ACT_FLAG_OFFSET + 1;
/// `copy_above_palette_indices_flag` and
/// `copy_above_indices_for_final_run_flag` (Table 9-41): one context, which
/// Table 9-4 assigns to both.
pub const COPY_ABOVE_FLAG_OFFSET: usize = PALETTE_RUN_PREFIX_OFFSET + 8;
/// `palette_transpose_flag` (Table 9-42): one context.
pub const PALETTE_TRANSPOSE_FLAG_OFFSET: usize = COPY_ABOVE_FLAG_OFFSET + 1;
/// Every context, the SCC ones included.
pub const NUM_CONTEXTS_ALL: usize = PALETTE_TRANSPOSE_FLAG_OFFSET + 1;
/// `initValue` of every SCC context, for every `initType` (Tables 9-38 to
/// 9-42).
const SCC_INIT_VALUE: u8 = 154;

/// The full context state of a slice / substream.
#[derive(Clone)]
pub struct Contexts {
    /// One byte per context.
    pub c: [Ctx; NUM_CONTEXTS_ALL],
    /// `StatCoeff` (persistent Rice adaptation; unused without the range
    /// extension but kept with the state so WPP storage is uniform).
    pub stat_coeff: [u8; 4],
    /// The palette predictor, `PredictorPaletteEntries[comp][i]` stored
    /// entry-major with `PredictorPaletteSize` its length. It is stored and
    /// synchronised exactly when the contexts are (9.3.1), so it lives with
    /// them; empty (and never allocated) without palette mode.
    pub palette: Vec<[u16; 3]>,
}

impl Contexts {
    /// Initialise for `init_type` (0: I, 1: P, 2: B after cabac_init_flag
    /// swap) at `SliceQpY`.
    pub fn new(init_type: usize, qp: i32) -> Self {
        let mut c = [0u8; NUM_CONTEXTS_ALL];
        for (i, v) in c.iter_mut().enumerate().take(NUM_CONTEXTS) {
            *v = init_ctx_hevc(CABAC_INIT[init_type][i], qp);
        }
        for v in c.iter_mut().skip(NUM_CONTEXTS) {
            *v = init_ctx_hevc(SCC_INIT_VALUE, qp);
        }
        Contexts {
            c,
            stat_coeff: [0; 4],
            palette: Vec::new(),
        }
    }
}
