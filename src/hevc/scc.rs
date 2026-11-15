//! The screen content coding extensions (H.265 V11, SCC): palette mode
//! (7.3.8.13, 8.4.4.2.7), intra block copy as prediction from the current
//! picture (`pps_curr_pic_ref_enabled_flag`, 8.3.4 / 8.5.3.2), and the
//! adaptive colour transform of residuals (8.6.8). Adaptive motion vector
//! resolution lives in the prediction unit (8.5.3.2.1 / 8.5.3.2.2), the
//! parameter-set and slice-header syntax in their own modules.

use crate::cabac::Cabac;
use crate::{Error, Result};

use super::ctu::{SliceDec, chroma_qp};
use super::ctx::*;
use super::frame::{Frame, Sample};
use super::pps::Pps;
use super::sps::Sps;

/// The largest palette coding block (palette CUs are no larger than the
/// largest transform block, 32x32).
const MAX_PALETTE_CB: usize = 32 * 32;

/// What a slice segment's decoding of the SCC tools keeps beside the rest
/// of [`SliceDec`].
pub struct SccDec<'a, S: Sample> {
    /// The current picture's samples before the in-loop filters, when it
    /// may be referenced by its own slices: each coding unit is copied here
    /// once reconstructed, and prediction from the current picture reads
    /// it (8.1.3, "the current decoded picture before the invocation of the
    /// in-loop filter process").
    pub ibc: Option<&'a mut Frame<S>>,
    /// Per list, per reference index: the entry is the current picture.
    pub is_cur: [Vec<bool>; 2],
    /// `TwoVersionsOfCurrDecPicFlag` (7-40).
    pub two_versions: bool,
    /// The current TU is under the adaptive colour transform: its three
    /// residuals are collected in `act_res` instead of being added.
    pub act_capture: bool,
    /// See `act_capture`: Y, Cb, Cr residuals of the TU, row-major.
    pub act_res: [Vec<i32>; 3],
    /// Palette scratch.
    pub(crate) palette: PaletteScratch,
}

impl<S: Sample> Default for SccDec<'_, S> {
    fn default() -> Self {
        SccDec {
            ibc: None,
            is_cur: [Vec::new(), Vec::new()],
            two_versions: false,
            act_capture: false,
            act_res: [Vec::new(), Vec::new(), Vec::new()],
            palette: PaletteScratch::default(),
        }
    }
}

impl<S: Sample> SccDec<'_, S> {
    /// Whether `RefPicList{list}[ref_idx]` is the current picture.
    #[inline]
    pub fn is_cur(&self, list: usize, ref_idx: usize) -> bool {
        self.is_cur[list].get(ref_idx).copied().unwrap_or(false)
    }
}

/// Buffers for one palette coding unit, in the scan domain (before
/// `palette_transpose_flag`), `nCbS` samples per row.
#[derive(Default)]
pub(crate) struct PaletteScratch {
    /// `PaletteIndexMap`.
    index: Vec<u8>,
    /// `CopyAboveIndicesFlag`.
    copy_above: Vec<bool>,
    /// `PaletteEscapeVal[cIdx]`.
    escape: [Vec<u16>; 3],
    /// `PaletteIndexIdc`.
    idc: Vec<u32>,
}

/// 9.3.2.3: the palette predictor at the start of a slice, tile or
/// unsynchronised WPP row — the PPS initializers, else the SPS ones, else
/// empty.
pub fn palette_predictor_init(sps: &Sps, pps: &Pps) -> Vec<[u16; 3]> {
    if !sps.palette_mode() {
        return Vec::new();
    }
    if let Some(init) = pps.scc.as_ref().and_then(|s| s.palette_init.as_ref()) {
        return init.clone();
    }
    sps.scc
        .as_ref()
        .and_then(|s| s.palette_init.clone())
        .unwrap_or_default()
}

/// The SCC configurations this decoder refuses, by name: the tools
/// combined with what no conformance stream exercises alongside them, where
/// the combination takes code paths of its own (and, for palette mode in
/// 4:2:2, where 8-69 / 8-70 do not say which index-map sample a chroma
/// sample reads). Every other configuration of the tools decodes the JCT-VC
/// SCC suite bit-exactly.
pub fn unsupported(sps: &Sps, pps: &Pps) -> Option<&'static str> {
    let cat = sps.chroma_array_type();
    let palette = sps.palette_mode();
    let ibc = pps.curr_pic_ref();
    let act = pps.act();
    let (bdy, bdc) = (sps.bit_depth_luma, sps.bit_depth_chroma);
    if palette && cat == 2 {
        return Some("palette mode in 4:2:2");
    }
    if palette && cat == 0 {
        return Some("palette mode in 4:0:0");
    }
    if ibc && (cat == 0 || cat == 2) {
        return Some("prediction from the current picture in 4:0:0 / 4:2:2");
    }
    if (palette || act) && (bdy != bdc || bdy > 10) {
        return Some(
            "palette mode / adaptive colour transform above 10 bits or at unequal bit depths",
        );
    }
    if (palette || act) && pps.transquant_bypass_enabled {
        return Some("palette mode / adaptive colour transform with cu_transquant_bypass");
    }
    if palette && pps.dependent_slice_segments_enabled {
        return Some("palette predictor across dependent slice segments");
    }
    if ibc && (pps.weighted_pred || pps.weighted_bipred) {
        return Some("weighted prediction with the current picture as a reference");
    }
    if ibc && pps.lists_modification_present {
        return Some("reference list modification with the current picture as a reference");
    }
    if ibc && pps.constrained_intra_pred {
        return Some("constrained intra prediction with the current picture as a reference");
    }
    None
}

/// k-th order Exp-Golomb, all bins bypass (9.3.3.3).
fn eg_k(c: &mut Cabac, mut k: u32) -> Result<u32> {
    let mut v = 0u32;
    while c.bypass() != 0 {
        v = v.wrapping_add(1 << k);
        k += 1;
        if k > 31 {
            return Err(Error::bitstream("Exp-Golomb prefix runaway"));
        }
    }
    Ok(v.wrapping_add(c.bypass_bits_wide(k)))
}

/// Truncated binary with `cmax`, all bins bypass (9.3.3.6).
fn truncated_binary(c: &mut Cabac, cmax: u32) -> u32 {
    let n = cmax + 1;
    let k = 31 - n.leading_zeros();
    let u = (1u32 << (k + 1)) - n;
    let v = c.bypass_bits_wide(k);
    if v < u {
        v
    } else {
        ((v << 1) | c.bypass()) - u
    }
}

/// The traverse scan (6.5.6) position `pos` of an `n x n` block, `(x, y)`.
#[inline]
fn trav(pos: usize, log2: u32) -> (usize, usize) {
    let n = 1usize << log2;
    let y = pos >> log2;
    let x = pos & (n - 1);
    (if y & 1 == 0 { x } else { n - 1 - x }, y)
}

impl<S: Sample> SliceDec<'_, S> {
    /// `palette_coding()` (7.3.8.13) of the CU at `(x0, y0)`, and its
    /// reconstruction (8.4.4.2.7, 8.4.1) and predictor update (8-79).
    pub(crate) fn palette_coding(
        &mut self,
        x0: i32,
        y0: i32,
        log2_cb: u32,
        bypass: bool,
    ) -> Result<()> {
        let scc = self
            .sps
            .scc
            .as_ref()
            .expect("palette mode needs the SCC extension");
        let max_size = scc.palette_max_size as usize;
        let max_pred_size = scc.palette_max_predictor_size as usize;
        let cat = self.sps.chroma_array_type();
        let num_comps = if cat == 0 { 1 } else { 3 };
        let n = 1usize << log2_cb;
        let nn = n * n;
        let pred = std::mem::take(&mut self.cx.palette);

        // Reuse flags from palette_predictor_run (EG0).
        let mut reuse = [false; 256];
        let mut num_predicted = 0usize;
        let mut i = 0usize;
        while i < pred.len() && num_predicted < max_size {
            let run = eg_k(&mut self.cabac, 0)? as usize;
            if run == 1 {
                break;
            }
            if run > 1 {
                i += run - 1;
            }
            if i >= pred.len() {
                return Err(Error::bitstream(
                    "palette_predictor_run beyond the predictor",
                ));
            }
            reuse[i] = true;
            num_predicted += 1;
            i += 1;
        }
        let num_signalled = if num_predicted < max_size {
            eg_k(&mut self.cabac, 0)? as usize
        } else {
            0
        };
        let size = num_predicted + num_signalled;
        if size > max_size {
            return Err(Error::bitstream(
                "CurrentPaletteSize beyond palette_max_size",
            ));
        }
        // CurrentPaletteEntries (7-82): the reused entries in predictor
        // order, then the signalled ones (component by component).
        let mut palette: Vec<[u16; 3]> = pred
            .iter()
            .zip(reuse.iter())
            .filter(|(_, r)| **r)
            .map(|(e, _)| *e)
            .collect();
        palette.resize(size, [0; 3]);
        for c in 0..num_comps {
            let bd = if c == 0 {
                self.sps.bit_depth_luma
            } else {
                self.sps.bit_depth_chroma
            };
            for e in palette[num_predicted..].iter_mut() {
                e[c] = self.cabac.bypass_bits_wide(bd) as u16;
            }
        }
        let escape_present = if size != 0 {
            self.cabac.bypass() != 0
        } else {
            true
        };
        let max_index = size as i32 - 1 + escape_present as i32;

        let mut sc = std::mem::take(&mut self.scc.palette);
        sc.idc.clear();
        let mut num_indices = 1usize;
        let mut final_run_copy = false;
        let mut transpose = false;
        if max_index > 0 {
            // num_palette_indices_minus1 (9.3.3.14): TR prefix + EG(k+1).
            let rice = 3 + ((max_index as u32 + 1) >> 3);
            let mut prefix = 0u32;
            while prefix < 4 && self.cabac.bypass() != 0 {
                prefix += 1;
            }
            let v = if prefix < 4 {
                (prefix << rice) + self.cabac.bypass_bits_wide(rice)
            } else {
                (4u32 << rice) + eg_k(&mut self.cabac, rice + 1)?
            };
            num_indices = v as usize + 1;
            if num_indices > nn {
                return Err(Error::bitstream("num_palette_indices_minus1 out of range"));
            }
            for k in 0..num_indices {
                // palette_idx_idc: TB, cMax MaxPaletteIndex for the first,
                // MaxPaletteIndex - 1 after (9.3.3.13); absent (0) when
                // that is 0.
                let cmax = max_index as u32 - (k > 0) as u32;
                sc.idc.push(if cmax > 0 {
                    truncated_binary(&mut self.cabac, cmax)
                } else {
                    0
                });
            }
            final_run_copy = bin(&mut self.cabac, &mut self.cx, COPY_ABOVE_FLAG_OFFSET);
            transpose = bin(&mut self.cabac, &mut self.cx, PALETTE_TRANSPOSE_FLAG_OFFSET);
        } else {
            sc.idc.push(0);
        }
        if escape_present {
            self.delta_qp(x0, y0)?;
            if !bypass {
                self.chroma_qp_offset();
            }
        }

        // The index map, in runs (7.3.8.13, 7.4.9.13).
        sc.index.clear();
        sc.index.resize(nn, 0);
        sc.copy_above.clear();
        sc.copy_above.resize(nn, false);
        let at = |(x, y): (usize, usize)| y * n + x;
        let mut remaining = num_indices;
        let mut pos = 0usize;
        while pos < nn {
            let cur = trav(pos, log2_cb);
            let prev = if pos > 0 {
                at(trav(pos - 1, log2_cb))
            } else {
                0
            };
            let mut run_minus1 = nn - pos - 1;
            let mut copy_above = false;
            if max_index > 0 && pos >= n && !sc.copy_above[prev] {
                copy_above = if remaining > 0 && pos < nn - 1 {
                    bin(&mut self.cabac, &mut self.cx, COPY_ABOVE_FLAG_OFFSET)
                } else {
                    !(pos == nn - 1 && remaining > 0)
                };
            }
            let mut idc = 0u32;
            let mut index = 0u32;
            if !copy_above {
                let k = num_indices - remaining;
                idc = *sc
                    .idc
                    .get(k)
                    .ok_or_else(|| Error::bitstream("palette indices exhausted"))?;
                // 7-83 / 7-84: the coded index skips the one the run could
                // not have continued with.
                let adjusted_ref = if pos == 0 {
                    max_index as u32 + 1
                } else if !sc.copy_above[prev] {
                    sc.index[prev] as u32
                } else {
                    sc.index[at(cur) - n] as u32
                };
                index = idc + (idc >= adjusted_ref) as u32;
                if index as i32 > max_index {
                    return Err(Error::bitstream("palette index beyond MaxPaletteIndex"));
                }
            }
            if max_index > 0 {
                if !copy_above {
                    remaining -= 1;
                }
                if remaining > 0 || copy_above != final_run_copy {
                    let max_run_minus1 =
                        (nn - pos - 1) as i64 - remaining as i64 - final_run_copy as i64;
                    if max_run_minus1 < 0 {
                        return Err(Error::bitstream("PaletteMaxRunMinus1 negative"));
                    }
                    run_minus1 = if max_run_minus1 > 0 {
                        self.palette_run(copy_above, idc, max_run_minus1 as u32) as usize
                    } else {
                        0
                    };
                }
            }
            if pos + run_minus1 >= nn {
                return Err(Error::bitstream("palette run beyond the block"));
            }
            for _ in 0..=run_minus1 {
                let p = at(trav(pos, log2_cb));
                sc.copy_above[p] = copy_above;
                sc.index[p] = if copy_above {
                    if p < n {
                        return Err(Error::bitstream("palette copy-above in the first row"));
                    }
                    sc.index[p - n]
                } else {
                    index as u8
                };
                pos += 1;
            }
        }

        // Escape samples, component by component in scan order.
        if escape_present {
            for c in 0..num_comps {
                sc.escape[c].clear();
                sc.escape[c].resize(nn, 0);
                let bd = if c == 0 {
                    self.sps.bit_depth_luma
                } else {
                    self.sps.bit_depth_chroma
                };
                for spos in 0..nn {
                    let (x, y) = trav(spos, log2_cb);
                    if sc.index[at((x, y))] as i32 != max_index {
                        continue;
                    }
                    let coded = c == 0
                        || cat == 3
                        || (cat == 1 && x % 2 == 0 && y % 2 == 0)
                        || (cat == 2 && !transpose && x % 2 == 0)
                        || (cat == 2 && transpose && y % 2 == 0);
                    if coded {
                        let v = if bypass {
                            self.cabac.bypass_bits_wide(bd)
                        } else {
                            eg_k(&mut self.cabac, 3)?
                        };
                        sc.escape[c][at((x, y))] = v.min(u16::MAX as u32) as u16;
                    }
                }
            }
        }

        // Reconstruction (8.4.4.2.7, written through 8-13 / 8-14 and
        // 8-17 – 8-20): a picture sample (x, y) of component c reads the
        // scan-domain map at (x * SubWidth, y * SubHeight), transposed when
        // palette_transpose_flag is set.
        let (sw, sh) = self.sps.sub_wh();
        for c in 0..num_comps {
            let (csw, csh) = if c == 0 { (1, 1) } else { (sw, sh) };
            let (w, h) = (n / csw, n / csh);
            let bd = if c == 0 {
                self.sps.bit_depth_luma
            } else {
                self.sps.bit_depth_chroma
            };
            let qp = self.escape_qp(c);
            let max = (1i64 << bd) - 1;
            let plane = match c {
                0 => &mut self.frame.y,
                1 => &mut self.frame.cb,
                _ => &mut self.frame.cr,
            };
            let (px, py) = (x0 as usize / csw, y0 as usize / csh);
            let (vis_w, vis_h) = (
                w.min(plane.width.saturating_sub(px)),
                h.min(plane.height.saturating_sub(py)),
            );
            let stride = plane.stride;
            let base = plane.offset(px as isize, py as isize);
            for y in 0..vis_h {
                for x in 0..vis_w {
                    let (lx, ly) = (x * csw, y * csh);
                    let p = if transpose { lx * n + ly } else { ly * n + lx };
                    let idx = sc.index[p] as usize;
                    let v = if escape_present && idx as i32 == max_index {
                        let e = sc.escape[c][p] as i64;
                        if bypass {
                            e
                        } else {
                            // 8-77 / 8-78.
                            const LEVEL_SCALE: [i64; 6] = [40, 45, 51, 57, 64, 72];
                            (((e * LEVEL_SCALE[(qp % 6) as usize]) << (qp / 6)) + 32) >> 6
                        }
                        .clamp(0, max)
                    } else {
                        palette[idx][c] as i64
                    };
                    plane.data[base + y * stride + x] = S::from_i32(v as i32);
                }
            }
        }

        // Predictor update (8-79): the current palette, then the unused
        // predictor entries, up to PaletteMaxPredictorSize.
        let mut next = palette;
        for (e, r) in pred.iter().zip(reuse.iter()) {
            if next.len() >= max_pred_size {
                break;
            }
            if !*r {
                next.push(*e);
            }
        }
        next.truncate(max_pred_size);
        self.cx.palette = next;
        self.scc.palette = sc;
        debug_assert!(nn <= MAX_PALETTE_CB);
        Ok(())
    }

    /// `palette_run_prefix` / `palette_run_suffix` → `PaletteRunMinus1`
    /// (7-85 / 7-86), with the contexts of 9.3.4.2.8.
    fn palette_run(&mut self, copy_above: bool, idc: u32, max_run_minus1: u32) -> u32 {
        // Prefix: TR, cMax = Floor(Log2(PaletteMaxRunMinus1)) + 1.
        let cmax = (31 - max_run_minus1.leading_zeros()) + 1;
        let mut prefix = 0u32;
        while prefix < cmax {
            let b = if prefix > 4 {
                self.cabac.bypass()
            } else {
                let inc = if copy_above {
                    [5, 6, 6, 7, 7][prefix as usize]
                } else if prefix == 0 {
                    if idc < 1 {
                        0
                    } else if idc < 3 {
                        1
                    } else {
                        2
                    }
                } else {
                    [0, 3, 3, 4, 4][prefix as usize]
                };
                bin(
                    &mut self.cabac,
                    &mut self.cx,
                    PALETTE_RUN_PREFIX_OFFSET + inc,
                ) as u32
            };
            if b == 0 {
                break;
            }
            prefix += 1;
        }
        if prefix < 2 {
            return prefix;
        }
        let offset = 1u32 << (prefix - 1);
        if max_run_minus1 == offset {
            return offset;
        }
        // Suffix: TB.
        let cmax = if (offset << 1) > max_run_minus1 {
            max_run_minus1 - offset
        } else {
            offset - 1
        };
        offset + truncated_binary(&mut self.cabac, cmax)
    }

    /// The `qP` of an escape sample of component `c` (8-73 – 8-75):
    /// `Max(0, Qp′Y)` or the chroma `Qp′Cb` / `Qp′Cr`.
    fn escape_qp(&self, c: usize) -> i32 {
        if c == 0 {
            (self.qp_y + 6 * (self.sps.bit_depth_luma as i32 - 8)).max(0)
        } else {
            let bd_off_c = 6 * (self.sps.bit_depth_chroma as i32 - 8);
            let off = if c == 1 {
                self.pps.cb_qp_offset + self.hdr.cb_qp_offset
            } else {
                self.pps.cr_qp_offset + self.hdr.cr_qp_offset
            };
            let qpi = (self.qp_y + off + self.cu_qp_offset_c[c - 1]).clamp(-bd_off_c, 57);
            (chroma_qp(self.sps.chroma_array_type(), qpi) + bd_off_c).max(0)
        }
    }

    /// Copy a reconstructed coding unit (`w x h` luma samples at `(x, y)`,
    /// already clipped to the picture) into the current picture's unfiltered
    /// copy, which prediction from the current picture reads.
    pub(crate) fn ibc_store(&mut self, x: usize, y: usize, w: usize, h: usize) {
        let Some(dst) = self.scc.ibc.as_deref_mut() else {
            return;
        };
        let (sw, sh) = self.sps.sub_wh();
        let mono = self.sps.chroma_array_type() == 0;
        let src: &Frame<S> = &*self.frame;
        for c in 0..if mono { 1 } else { 3 } {
            let (csw, csh) = if c == 0 { (1, 1) } else { (sw, sh) };
            let (sp, dp) = match c {
                0 => (&src.y, &mut dst.y),
                1 => (&src.cb, &mut dst.cb),
                _ => (&src.cr, &mut dst.cr),
            };
            let (px, py, pw, ph) = (x / csw, y / csh, w / csw, h / csh);
            for yy in py..py + ph {
                let so = sp.offset(px as isize, yy as isize);
                let d_o = dp.offset(px as isize, yy as isize);
                dp.data[d_o..d_o + pw].copy_from_slice(&sp.data[so..so + pw]);
            }
        }
    }

    /// The adaptive colour transform of one TU's residuals (8.6.8.2) and
    /// their addition to the prediction: `n x n` at `(x, y)` in every
    /// component (4:4:4).
    pub(crate) fn act_reconstruct(&mut self, x: usize, y: usize, n: usize, bypass: bool) {
        let (bdy, bdc) = (self.sps.bit_depth_luma, self.sps.bit_depth_chroma);
        let range = |bd: u32| -> (i32, i32) {
            let log2 = self.sps.log2_transform_range(bd);
            (-(1i32 << log2), (1i32 << log2) - 1)
        };
        let (min_y, max_y) = range(bdy);
        let (min_c, max_c) = range(bdc);
        let bd_max = bdy.max(bdc);
        let (dy, dc) = ((bd_max - bdy) as i32, (bd_max - bdc) as i32);
        let off_y = if dy > 0 { 1 << (dy - 1) } else { 0 };
        let off_c = if dc > 0 { 1 << (dc - 1) } else { 0 };
        let [ry, rcb, rcr] = &mut self.scc.act_res;
        for ((a, b), c) in ry.iter_mut().zip(rcb.iter_mut()).zip(rcr.iter_mut()) {
            let (mut y0, mut cb, mut cr) = (
                (*a).clamp(min_y, max_y) as i64,
                (*b).clamp(min_c, max_c) as i64,
                (*c).clamp(min_c, max_c) as i64,
            );
            if !bypass {
                y0 <<= dy;
                cb <<= dc + 1;
                cr <<= dc + 1;
            }
            let tmp = y0 - (cb >> 1);
            y0 = tmp + cb;
            cb = tmp - (cr >> 1);
            cr += cb;
            if !bypass {
                y0 = (y0 + off_y) >> dy;
                cb = (cb + off_c) >> dc;
                cr = (cr + off_c) >> dc;
            }
            *a = y0 as i32;
            *b = cb as i32;
            *c = cr as i32;
        }
        for c in 0..3 {
            let bd = if c == 0 { bdy } else { bdc };
            let max = (1i32 << bd) - 1;
            let plane = match c {
                0 => &mut self.frame.y,
                1 => &mut self.frame.cb,
                _ => &mut self.frame.cr,
            };
            let stride = plane.stride;
            let off = plane.offset(x as isize, y as isize);
            if self.wide {
                crate::dsp::hevc::add_residual_wide(
                    &mut plane.data[off..],
                    stride,
                    &self.scc.act_res[c],
                    n,
                    max,
                );
            } else {
                let mut coeffs = std::mem::take(&mut self.coeffs);
                if coeffs.len() < n * n {
                    coeffs.resize(1024, 0);
                }
                for (d, &v) in coeffs.iter_mut().zip(&self.scc.act_res[c]) {
                    *d = v.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                }
                (self.dsp.add_residual)(&mut plane.data[off..], stride, &coeffs, n, max);
                self.coeffs = coeffs;
            }
        }
    }
}

#[inline(always)]
fn bin(c: &mut Cabac, cx: &mut Contexts, ctx: usize) -> bool {
    c.decision(&mut cx.c[ctx]) != 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitwriter::BitWriter;
    use crate::hevc::pps::PpsScc;
    use crate::hevc::sps::SpsScc;

    /// A PPS RBSP with every tool off and the extensions `ext` (range,
    /// multilayer, 3D, SCC) present, the SCC one written by `scc`.
    fn pps_rbsp(ext: [bool; 4], scc: impl Fn(&mut BitWriter)) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.ue(0); // pps_pic_parameter_set_id
        w.ue(0); // pps_seq_parameter_set_id
        w.flag(false); // dependent_slice_segments_enabled_flag
        w.flag(false); // output_flag_present_flag
        w.bits(3, 0); // num_extra_slice_header_bits
        w.flag(false); // sign_data_hiding_enabled_flag
        w.flag(false); // cabac_init_present_flag
        w.ue(0); // num_ref_idx_l0_default_active_minus1
        w.ue(0); // num_ref_idx_l1_default_active_minus1
        w.se(0); // init_qp_minus26
        w.flag(false); // constrained_intra_pred_flag
        w.flag(false); // transform_skip_enabled_flag
        w.flag(false); // cu_qp_delta_enabled_flag
        w.se(0); // pps_cb_qp_offset
        w.se(0); // pps_cr_qp_offset
        w.flag(false); // pps_slice_chroma_qp_offsets_present_flag
        w.flag(false); // weighted_pred_flag
        w.flag(false); // weighted_bipred_flag
        w.flag(false); // transquant_bypass_enabled_flag
        w.flag(false); // tiles_enabled_flag
        w.flag(false); // entropy_coding_sync_enabled_flag
        w.flag(false); // pps_loop_filter_across_slices_enabled_flag
        w.flag(false); // deblocking_filter_control_present_flag
        w.flag(false); // pps_scaling_list_data_present_flag
        w.flag(false); // lists_modification_present_flag
        w.ue(0); // log2_parallel_merge_level_minus2
        w.flag(false); // slice_segment_header_extension_present_flag
        w.flag(true); // pps_extension_present_flag
        for f in ext {
            w.flag(f);
        }
        w.bits(4, 0); // pps_extension_4bits
        if ext[0] {
            // pps_range_extension() with nothing on.
            w.flag(false); // cross_component_prediction_enabled_flag
            w.flag(false); // chroma_qp_offset_list_enabled_flag
            w.ue(0); // log2_sao_offset_scale_luma
            w.ue(0); // log2_sao_offset_scale_chroma
        }
        scc(&mut w);
        w.rbsp_trailing_bits();
        w.into_rbsp()
    }

    fn scc_ext(w: &mut BitWriter) {
        w.flag(true); // pps_curr_pic_ref_enabled_flag
        w.flag(true); // residual_adaptive_colour_transform_enabled_flag
        w.flag(true); // pps_slice_act_qp_offsets_present_flag
        w.se(2); // pps_act_y_qp_offset_plus5
        w.se(-1); // pps_act_cb_qp_offset_plus5
        w.se(1); // pps_act_cr_qp_offset_plus3
        w.flag(true); // pps_palette_predictor_initializers_present_flag
        w.ue(2); // pps_num_palette_predictor_initializers
        w.flag(false); // monochrome_palette_flag
        w.ue(0); // luma_bit_depth_entry_minus8
        w.ue(0); // chroma_bit_depth_entry_minus8
        for v in [10, 20, 30, 40, 50, 60] {
            w.bits(8, v); // component by component
        }
    }

    #[test]
    fn the_pps_scc_extension_is_read_behind_the_range_extension() {
        let pps = Pps::parse(&pps_rbsp([true, false, false, true], scc_ext)).expect("parses");
        assert!(pps.range_ext);
        assert_eq!(
            pps.scc,
            Some(PpsScc {
                curr_pic_ref: true,
                act: true,
                slice_act_qp_offsets_present: true,
                act_qp_offsets: [-3, -6, -2],
                palette_init: Some(vec![[10, 30, 50], [20, 40, 60]]),
            })
        );
        assert!(pps.curr_pic_ref() && pps.act());
    }

    #[test]
    fn an_scc_extension_behind_an_unparsed_one_is_refused_not_misread() {
        for ext in [[false, true, false, true], [false, false, true, true]] {
            let e = Pps::parse(&pps_rbsp(ext, scc_ext)).expect_err("refused");
            assert!(matches!(e, Error::Unsupported(_)), "{e}");
        }
        // Without an SCC extension behind it, the multilayer one is
        // skipped as before (the base layer does not need it).
        let pps = Pps::parse(&pps_rbsp([false, true, false, false], |w| {
            w.flag(true); // some multilayer extension bit
        }))
        .expect("parses");
        assert_eq!(pps.scc, None);
    }

    #[test]
    fn unverified_combinations_are_refused_by_name() {
        use crate::encode::Config;
        use crate::encode::h265_syntax::{Geometry, write_sps};
        use crate::nal::unescape_rbsp;
        let cfg = Config {
            width: 64,
            height: 64,
            ..Config::default()
        };
        let g = Geometry::new(&cfg);
        let mut sps = Sps::parse(&unescape_rbsp(&write_sps(&cfg, &g, 8, None)))
            .expect("the encoder's SPS parses");
        let mut pps = Pps::parse(&pps_rbsp([false, false, false, true], scc_ext)).unwrap();
        sps.scc = Some(SpsScc {
            curr_pic_ref: true,
            palette_mode: true,
            palette_max_size: 63,
            palette_max_predictor_size: 128,
            ..SpsScc::default()
        });
        sps.chroma_format_idc = 3;
        assert_eq!(unsupported(&sps, &pps), None);
        sps.chroma_format_idc = 2;
        assert_eq!(unsupported(&sps, &pps), Some("palette mode in 4:2:2"));
        sps.chroma_format_idc = 3;
        pps.weighted_pred = true;
        assert_eq!(
            unsupported(&sps, &pps),
            Some("weighted prediction with the current picture as a reference")
        );
        pps.weighted_pred = false;
        sps.bit_depth_chroma = 10;
        assert!(unsupported(&sps, &pps).is_some());
    }
}
