# rivet-h26x

[![crates.io](https://img.shields.io/crates/v/rivet-h26x.svg)](https://crates.io/crates/rivet-h26x)
[![docs.rs](https://docs.rs/rivet-h26x/badge.svg)](https://docs.rs/rivet-h26x)
[![CI](https://github.com/safewords/rivet-h26x-codecs/actions/workflows/ci.yml/badge.svg)](https://github.com/safewords/rivet-h26x-codecs/actions/workflows/ci.yml)

Native **H.264/AVC** and **H.265/HEVC** decoders in Rust: no C, no system
libraries, no build script, nothing to install on a build host. Bit-exact
against the JVT and JCT-VC conformance suites, frame- and wavefront-threaded,
with a run-time ladder of SSE2-through-AVX2 and NEON kernels, and a
supplementary AVX-512 tier over AVX2 for the shapes where 512-bit lanes pay.

The SIMD is written in Rust intrinsics, not assembly, with one exception:
AArch64's `sdot` has no intrinsic on stable Rust yet
([rust-lang/rust#117224](https://github.com/rust-lang/rust/issues/117224)), so
that one instruction goes through a stable `asm!` wrapper carrying the
signature the intrinsic will have. Swapping it back is a one-line change.

Written for the **[rivet](https://github.com/safewords/rivet)**
transcoder, where they are the software decode tier for the two codecs every
camera, phone and broadcast chain emits — under the GPU decoders (NVDEC / AMF /
QSV) so a machine without a usable GPU still decodes — and usable on their own
by anything that has Annex-B NAL units and wants planar pictures back.

Published as `rivet-h26x`; **imported as `h26x`** (`use h26x::…`). One
dependency (`thiserror`), no features, no build script.

```toml
[dependencies]
h26x = { package = "rivet-h26x", version = "0.2" }
```

## What it decodes

| | supported | refused with `Error::Unsupported` |
|---|---|---|
| **H.264** | Baseline / Main / Extended / High / High 10 / High 4:2:2 / High 4:4:4 Predictive / CAVLC 4:4:4 Intra (and the Intra profiles): frames, field pictures (PAFF — field / frame reference lists and marking, field POCs, colocated field / frame mapping) and MBAFF (macroblock-adaptive frame / field: pair-wise neighbour derivation, field-scan and field-context entropy coding, mixed frame / field prediction and direct-mode mapping, mixed-edge deblocking), 4:0:0 / 4:2:0 / 4:2:2 / 4:4:4 at 8–14-bit with equal or unequal luma / chroma depths, separate colour planes, lossless (transform bypass), CAVLC + CABAC, I/P/B, spatial + temporal direct, explicit + implicit weighting, 8x8 transform, scaling matrices, PCM, MMCO, multi-slice in any order (ASO), slice groups (FMO, all seven map types), slice data partitioning (partitions A / B / C, Extended profile), SP / SI slices (Extended profile), frame-num gaps, all three POC types, deblocking, VUI reorder hints, the old-x264 4:4:4 CABAC quirk | SP / SI outside the Extended profile's shape (CABAC, 4:2:2 / 4:4:4, > 8-bit, 8x8 transform) — combinations no profile allows: SP and SI exist only in the Extended profile (A.2.3), whose SPS has no chroma format or depth and whose PPS no CABAC or 8x8 transform, and every other profile allows I, P and B slices only. Bit depths above 14 |
| **H.265** | Main / Main 10 / Main 12, the format range extensions — Main 4:2:2 / 4:4:4 (10, 12, 16 Intra), the High Throughput 4:4:4 (16 Intra) and Monochrome profiles (4:0:0 / 4:2:0 / 4:2:2 / 4:4:4, 8–16-bit, unequal luma / chroma depths) — and the screen content coding extensions: CTB 16–64, AMP, transform skip (any size, rotation, single-context), scaling lists, sign hiding, PCM, `cu_transquant_bypass` (lossless), cu_qp_delta, chroma QP offset lists, cross-component prediction, implicit / explicit RDPCM, persistent Rice adaptation, high-precision weighted-prediction offsets, intra smoothing disabling, extended precision processing, CABAC bypass alignment, separate colour planes (three monochrome decodes joined into 4:4:4, 8.1.3), **SCC**: palette mode (predictor initialisers, escapes, transpose, WPP predictor sync), intra block copy (the current picture as a reference, predicted from its unfiltered samples), the adaptive colour transform (with its PPS / slice QP offsets), adaptive motion vector resolution (`use_integer_mv_flag`) and intra boundary filtering disabling; tiles, WPP (and both together), dependent slice segments, merge / AMVP / TMVP, explicit weighting, deblocking, SAO, long-term references, CRA / BLA / RASL handling, `pic_output_flag`, `no_output_of_prior_pics`, decoded-picture-hash SEI verification (`H26X_VERIFY_HASH=1`); the **base layer** of MV-HEVC / SHVC streams (every NAL unit with `nuh_layer_id` > 0 discarded, as a single-layer decoder does; a stream whose base layer is external outputs nothing) | Multi-layer output beyond the base layer (enhancement layers and views are discarded, not decoded). SCC combinations no conformance stream exercises: palette mode in 4:0:0 or 4:2:2 (where 8-69 / 8-70 do not say which index-map sample a 4:2:2 chroma sample reads), the current picture as a reference in 4:0:0 / 4:2:2, palette or ACT above 10 bits, at unequal depths or with `cu_transquant_bypass`, the palette predictor across dependent slice segments, weighted prediction / list modification / constrained intra prediction with the current picture as a reference, an SCC extension behind an unparsed multilayer / 3D parameter-set extension |

Both decoders are **bit-exact**. H.264 passes **all 204** JVT conformance
bitstreams (AVCv1 + FRExt — every profile set, every field-picture and MBAFF
stream, the three FMO / ASO streams and the two SP streams) against the
suite's reconstructed YUV (the JM reference decoder's output where the zip
ships none), and **all 38** JVT professional-profile bitstreams (High 10 /
4:2:2 / 4:4:4 Intra, CAVLC 4:4:4 Intra, High 4:4:4 Predictive at up to
14-bit, thirteen of them coded as separate colour planes), which ship no
reference output and are checked against the JM reference decoder. No JVT
zip carries slice data partitioning or unequal luma / chroma depths, so
**18 streams generated with the JM reference encoder** cover them (eight
partitioned — direct modes, weighting, constrained intra, fields, PAFF,
multiple slices; ten at unequal depths — every chroma format, both entropy
coders, the 8x8 transform, chroma weighted-prediction offsets, fields),
each matching the JM reference decoder frame for frame. Decoding is
deterministic across thread counts (every suite stream decoded on 1 and
12 threads gives the same bytes). It matches JM on the workspace fixtures too
(53 x264 / x265 encodes: CAVLC/CABAC, B-pyramids, weighting, 8x8, slices,
CQM, 10-bit, 4:2:2, 4:0:0, 4:4:4, lossless, interlaced). H.265 passes **147
of the 147** JCT-VC HEVC_v1 conformance bitstreams, **all 49** RExt
bitstreams and **all 15** SCC bitstreams (palette, intra block copy, the
adaptive colour transform, adaptive MV resolution, intra boundary filter
disabling; 4:2:0 and 4:4:4, 8 and 10-bit, RGB) against the suite's own
MD5s (HM's output where a zip ships none) — among them 16-bit, extended
precision, CABAC bypass alignment and unequal luma / chroma bit depths —
and decodes the base layer of **all 9** MV-HEVC and **all 70** SHVC
conformance bitstreams to the suite's layer-0 MD5 (HM's base-layer output
where the zip has none). No HEVC profile allows separate colour planes, so
nothing writes them; **4 generated streams** (three monochrome x265
encodes spliced into one separate-plane stream, 8 to 12-bit, multiple
slices, WPP, slices of the planes interleaved) decode to exactly HM's
decodes of the three monochrome streams. Every HM stream's
decoded-picture-hash SEI is checked as well, and it matches HM on the x265
feature fixtures.
The tools and the reference decoders are described in
[tools/README.md](tools/README.md); none of it involves FFmpeg. Bit depths above 12, and extended precision at any depth, decode on a
scalar `i32` pipeline beside the `i16` SIMD one the 8–12-bit streams keep.

`Unsupported` is a *classification*, not a failure: it names what the
stream uses that this decoder does not. In rivet, h26x is the last software
tier, so such a stream is refused with that reason rather than handed on.

## What it encodes

Encoders for both codecs are being built to the same standard as the
decoders, and the part that is settled is how they are verified, because an
encoder cannot have a conformance suite: a standard constrains what a
*decoder* must do with a bitstream and leaves an encoder free to choose any
legal one, so there is no golden output to compare against. Three properties
replace it, two of them exact:

1. **SELF** — the encoder records the reconstruction it believes its
   bitstream carries, and this crate's own decoder must reproduce it byte for
   byte. A mismatch is encoder/decoder state desync: always a bug, never a
   quality question, and it needs no reference data.
2. **CROSS** — the ITU-T reference decoder (JM for H.264, HM for H.265)
   must decode the bitstream to the same pictures our decoder does. SELF alone would pass if both of our sides shared a
   misreading; CROSS is what makes the output *legal* rather than merely
   self-compatible.
3. **QUALITY** — PSNR against the source, reported rather than gated, except
   in lossless mode where the reconstruction must equal the source exactly
   and the measurement becomes a check like the other two.

`tools/verify_encode.sh` gates 1 and 2 and reports 3, over seven generated
clips (4:0:0 through 4:4:4, plus a 50x34 one because cropping is a common
place to be wrong) and a configuration list that grows one axis at a time.

Current state: both encoders produce legal streams for every
configuration the gate lists (all-intra through IPB, both entropy coders
for H.264, 8 to 14 bits, 4:0:0 to 4:4:4, interlaced H.264); what is not
built refuses with `Error::Unsupported` naming the missing piece — the gate
distinguishes "not built" from "wrong".

### H.264 compression against x264

Progressive H.264 frame pictures are decided by a rate-distortion analysis
(`src/encode/h264_analyse.rs`) in three tiers, `--preset veryfast | medium |
slow` in `h26xenc` (`Config::preset`; the CLI preset also turns on three B
pictures, the 8x8 transform, sub-macroblock partitions except in veryfast,
the B pyramid and the lookahead; slow keeps five references):

- **Motion**: the three half-sample planes of each reference are built once
  (`h264_hpel.rs`), so every quarter-sample candidate is a plane read or one
  rounded average — bit-exact with the decoder's kernels at all sixteen
  positions (tested). Candidates are the median, skip and zero vectors, the
  coded neighbours' and the lookahead's half-resolution vectors; a hexagon
  (veryfast: small diamond) descent on SAD plus
  lambda times the vector's rate, then half- and quarter-sample refinement
  on SATD plus rate. P pictures search up to 3 (slow 5) references, and a
  P partition below 16x16 picks its own among those whose 16x16 cost came
  within 1.25x (slow 1.5x) of the best (mixed references). A B pair is
  refined jointly — each list's vector a quarter sample around with the
  other held, scored as the average they make (16x16: one round in
  veryfast, two otherwise; slow refines its partitions' pairs too), and
  list 1's search also starts from list 0's vector mirrored.
- **Decision**: shapes, B directions and intra modes are estimated at SATD
  plus rate; skip, the best one to three inter shapes at both transform
  sizes and the best intra type are then coded and compared on
  `SSD + lambda * R`, with `R` priced by the picture's own CABAC writer from
  its live context states (`CabacEncoder::estimating`) — no table of what
  syntax "usually" costs (`CabacEstimator`, the same writers compiled for
  pricing). Early skip when the skip prediction quantises to nothing and
  its error is small.
- **Quantisation**: the chosen macroblock is re-quantised by a trellis over
  each block's level choices, priced from the live contexts
  (`h264_trellis.rs`). Medium and slow quantise every candidate the
  comparison codes by the trellis, so their decisions see the levels they
  will write; veryfast skips the trellis in B pictures nothing references.
  B partitions below 16x16 are searched only when mixing the 16x16
  candidates per quadrant, unsearched, comes within 1.02x of the best
  whole-macroblock estimate (`AnalysisCfg::b_part_gate`, medium and slow).
- **Structure**: a B pyramid (the middle B a reference; a P picture's list 0
  reordered to its anchor), and a lookahead (`h264_lookahead.rs`) that
  propagates half-resolution intra/inter cost estimates back through each
  picture's vectors and codes reused reference macroblocks finer
  (`--tree N`), over frame quantiser offsets [I, P, B, Bref] = [-2, 0, 2, 1].

Measured by a parity script outside the repository: 30 frames of five clips, rivet at
QP 22/27/32/37 against x264 0.165 `--preset P --tune psnr` at CRF 18/23/28/33,
one thread, both decoded by ffmpeg and compared raw; BD-rate on PSNR-Y and
PSNR-avg (negative: rivet needs fewer bits). Time is the ratio of CPU seconds
of interleaved runs (rivet and x264 alternating, one encode at a time) on
one loaded Apple M-series machine, summed over the four rate points:

| `--preset` | clip | BD-rate Y | BD-rate avg | time vs x264 |
|---|---|---:|---:|---:|
| veryfast | foreman CIF | +3.27 % | +3.06 % | 2.19x |
| | mobile CIF | -6.82 % | -8.25 % | 2.60x |
| | akiyo CIF | -6.14 % | -5.84 % | 2.53x |
| | ducks_take_off 720p | -11.12 % | -13.29 % | 2.44x |
| | park_joy 720p | -2.58 % | -2.94 % | 2.50x |
| | **mean** | **-4.68 %** | **-5.45 %** | **2.46x** |
| medium | foreman CIF | +1.25 % | +1.30 % | 2.01x |
| | mobile CIF | +3.25 % | +2.69 % | 2.13x |
| | akiyo CIF | -4.46 % | -4.35 % | 2.02x |
| | ducks_take_off 720p | -1.18 % | -4.61 % | 2.22x |
| | park_joy 720p | +2.48 % | +2.58 % | 2.32x |
| | **mean** | **+0.27 %** | **-0.48 %** | **2.25x** |
| slow | foreman CIF | +2.87 % | +2.91 % | 1.74x |
| | mobile CIF | +3.51 % | +2.76 % | 1.40x |
| | akiyo CIF | -1.92 % | -1.63 % | 1.90x |
| | ducks_take_off 720p | +1.91 % | -2.59 % | 1.54x |
| | park_joy 720p | +2.43 % | +2.38 % | 1.53x |
| | **mean** | **+1.76 %** | **+0.77 %** | **1.54x** |

(Third round, 2026-11-15, rivet at --jobs 4 against the cached x264
times. The second round's tree, 0760d3b, timed in the same session:
-4.56 % / 2.96x, +1.94 % / 2.07x, +1.74 % / 2.01x.)

What the third round did, in instructions retired (`/usr/bin/time -l`,
deterministic where the loaded machine's seconds are not) summed over
the parity runs. Every change below up to the estimator is byte-identical
on all three presets; together they take 0760d3b's 176.6G / 376.0G /
874.0G instructions (veryfast / medium / slow) to 154.3G / 338.9G /
759.8G (-13 % / -10 % / -13 %; cycles at QP 27 -17 % / -14 % / -12 %):

| change | what it bought |
|---|---|
| CABAC residual writer: levels and significant positions gathered branch-free | veryfast -5 %, medium -2.5 % cycles |
| trellis: no 2 KB memset per block; levels scaled in raster order; zero-only positions as one lane expression | slow -4 % |
| search: no point scored twice (second sub-sample ring, the multi-hexagon's axis points) | medium -3 % instructions |
| NEON SATD: last butterfly stage folded into the sum (the absolute sum of a sum and a difference is twice the larger magnitude) | -0.7 % instructions |
| fused average + SATD for quarter-sample candidates (`H264EncDsp::satd_avg`, NEON) | slow -3 %, medium -2 % cycles |
| CABAC writer and estimator state held in registers through a block (`BinEnc::split`) | veryfast -2 % |
| CABAC writer: bins without a branch on their value | veryfast -3 % cycles |
| a reference's half-sample planes built once, not per picture searching it | veryfast -2.4 % instructions |
| block coders' residual a row at a time; intra modes scored in place | -2 % instructions |
| early skip: the error first, then quantisation without reconstruction, stopping at the first surviving level | veryfast -5.5 %, medium -3 % instructions |
| estimator total in fixed point (not byte-identical: near-ties in veryfast, BD-rate unchanged) | veryfast -2.4 %, medium -1.9 % cycles |

Then the decisions, against x264 at the same preset (BD-rate Y):

| change | veryfast | medium | slow |
|---|---:|---:|---:|
| after the speed work above | -4.56 % | +1.94 % | +1.74 % |
| B coded candidates at 0.85 lambda | -4.68 % | +1.71 % | +1.58 % |
| medium: intra analysed to 1.5x the best inter estimate (was 1.2x) | | +1.26 % | |
| medium: the trellis in every coded candidate | | +0.26 % (+23 % instructions) | |
| B partition gate 1.02 | | +0.27 % (-5 %) | |
| slow: hexagon instead of multi-hexagon, intra to 2.0x, gate | | | +1.76 % (-21 %) |

Tried and dropped in the third round (medium unless named; instructions
against the tree they were tried on): sub-partition search skipped below
an SATD threshold, +0.31 % for -8 %; B partitions off, +1.8 % for -17 %;
the gate at 1.0, +0.15 % for -9 % (kept at 1.02); one refinement round for
16x16, +0.24 % for -1.7 %; a quarter-sample diamond for partitions,
+0.57 % for -5 %; a half-sample diamond, +0.23 % for -3.7 %; SAD instead
of SATD at half samples, +0.1 % for -1.6 %; partitions below 16x16 from
their candidates without an integer descent, +0.1 % for -2.5 %; lambda
0.85x / 1.15x in P pictures, the trellis lambda 0.8x / 1.2x in B, no
better; intra to 2.0x in medium, -0.13 % for +6 %; rd_max 3 in medium,
-0.17 % for +4 %; slow with the gate at 1.0, +0.25 % for -14 %. Veryfast's
trades are all near 0.3 % BD-rate per 1 % of time (intra to 0.9x, +0.9 %
for -3 %; no pair refinement, +1.4 % for -4.5 %; no Intra_4x4, +1.6 % for
-5 %; no trellis, +5.7 % for -13 %), well off x264's own slope, so none
was taken. Speed attempts that changed nothing or cost: a branch-free
estimator significance map; the trellis's last position from an inverse
scan (more instructions); precomputed per-position level costs in the
trellis; LTO and one codegen unit (-2 %, and not what a dependent crate
builds). x264 black-box (medium): `--no-dct-decimate` +0.45 %, `--trellis
2` -1.6 % for +54 %, `--subme 8` -1.1 % for +14 %, `--me umh` -0.3 % for
+27 %.

The tree before this round (d66a816), timed the same way in the same
session: 3.92x (veryfast), 2.33x (medium), 1.70x (slow), at -2.44 / +5.37 /
+7.78 % BD-rate Y. Of the time, the changes that leave every stream
byte-identical (the CABAC writer and estimator, the trellis's tables, the
search's averaging; see "Encoder speed") take 0.82x / 0.84x / 0.86x of the
old time on their own; the compression changes then spend some of it back,
slow most (its trellis in every candidate is +27 %).

The starting point (2026-11-15, the placeholder decisions, `--bframes 3
--refs 4 --t8x8 --subparts`) was +57.5 % / +55.7 % at 3.69x against medium.
What each step bought against x264 medium (BD-rate Y, mean of five):

| step | BD-rate Y |
|---|---:|
| placeholder decisions | +57.5 % |
| rate-distortion analysis, half-sample planes, priced candidates | +24.8 % |
| B pyramid, lookahead propagation, frame offsets [I, P, B, Bref] = [0, 0, 3, 1] | +13.5 % |
| trellis from the live contexts | +8.4 % |
| lookahead vectors as search candidates | +7.9 % |
| lambda 0.8x, early skip held to an error ceiling | +6.4 % |
| three references for P pictures | +5.4 % |
| frame offsets [I, P, B, Bref] = [-2, 0, 2, 1] (were [0, 0, 3, 1]) | +4.0 % |
| mixed references in P partitions | +3.3 % |
| B pairs refined jointly (16x16, two rounds) | +2.0 % |
| list 0's vector mirrored as a list-1 start | +1.9 % |

The other tiers over the same steps: slow +7.78 → trellis in every
candidate +6.31 → offsets +4.97 → mixed references +4.08 → joint pair
refinement (16x16 and partitions) +2.44 → five references +1.80 → mirrored
start +1.74 %; veryfast -2.44 → offsets -4.03 → one round of pair
refinement -5.39 → no trellis in non-reference B pictures -4.50 (for 6 %
less time) → mirrored start -4.56 %.

Tried and dropped, with what they measured (same corpus, BD-rate Y against
medium unless named):

- (2026-11-15, second round) The trellis in every candidate in medium:
  -1.1 % for +25 % time. Partition pairs refined jointly in medium: -0.3 %
  for +9 %. RD refinement of a P 16x16 winner's vector (its eight
  quarter-sample neighbours coded and priced, slow): -0.15 % for +11 %.
  Medium without early skip: -0.4 % for +12 %. Six references in slow:
  -0.7 % for +10 % (five: -0.6 % for +5 %). A B frame offset of 1: a further
  -0.3 % for +3 %. Tree strength 1.0 / 1.25 / 1.75 / 2.0 under the new
  offsets: +0.4 to +1.4 % against 1.5. `lambda_ssd` 0.6x / 0.76x,
  `lambda_satd` from 0.65 / 1.1: -0.07 to +0.4 %.
- Veryfast trades: candidates at the 8x8 transform only, +3.2 % for -1 %
  time; no trellis, +6.6 % for -20 %; Intra_4x4 off in inter pictures,
  +1.6 % for -4 %; intra estimate ratio 0.8, +1.8 % for -4 %; no trellis on
  inter chroma, +0.07 % Y (+0.7 % avg) for -3 %; deciding on the estimates
  with the trellis on the winner, +28 % for -29 %.
- Speed attempts that saved nothing: a beam of two to four trellis paths;
  an integer all-zero prefilter before the trellis. Implicit B weighting
  changes no stream under the pyramid — every B picture sits midway
  between its two anchors, where the implicit weights are 32/32.

- Deciding veryfast on the SATD estimates alone and coding only the winner:
  +35.1 % against x264 veryfast at 2.0x; pricing one shape, +3.9 % at 2.9x;
  with the trellis, -2.4 % at 3.6x (kept).
- Lookahead propagation strength 1.0 / 2.0 / 3.0 against the kept 1.5
  (with the trellis): +8.50 / +9.51 % against +8.42 %; at 3.0 (before the
  trellis) +35.5 %. A half-sample lookahead search: +6.51 % against +6.41 %,
  dropped for its time.
- lambda_ssd 0.65x / 1.0x / 1.25x of the customary `0.85 * 2^((qp-12)/3)`:
  +6.80 / +7.19 / +8.45 % against 0.8x's +6.41 %; scaling the SATD lambda
  0.8x or 1.25x: no better.
- Trellis lambda 0.6x / 1.5x: +11.1 / +13.9 % against +10.3 % (before live
  contexts); the trellis priced from the slice's initial contexts rather
  than the live ones: +10.3 % against +8.5 %.
- Quarter-sample refinement on a diamond rather than the square: +0.8 % for
  5 % of the time. One refinement round instead of two: no time saved.
- Pricing at most one inter shape: +0.7 % for 0.83x the time; a 1.04
  instead of 1.1 estimate ratio: +0.35 % for 0.96x.
- Skipping intra unless its estimate beats the best inter one by 0.6x:
  +2.6 %. Forcing the 4x4 transform for inter: +2.5 %.
- Frame quantiser offsets without the lookahead [-3, 0, 2] / [-1, 0, 3] /
  [-2, 0, 4] / [0, 0, 4]: +24.8 / +24.6 / +23.6 / +24.5 % (placeholder era;
  the pyramid's [-3, 0, 2, 1] kept for streams without the lookahead).

Where the gap stands. x264's own features, measured black-box on this
corpus (BD-rate when x264 drops them): `--trellis 0` +4.6 %, `--ref 1`
+2.3 %, `--no-weightb` +1.2 %, `--subme 6` +1.2 %, `--no-mixed-refs`
+1.0 %; against slow, `--trellis 1` +2.1 %, `--subme 7` +1.5 %, `--ref 3`
+0.7 %; `--direct spatial`, `--b-adapt 0`, `--no-chroma-me` nothing.
Rivet now has the trellis (medium and slow: in every candidate), mixed and
multiple P references. What it lacks: more than one reference per B list
(and with it the implicit weighting x264's weightb applies between unequal
distances), which is most of what is left at medium and slow (~1 %);
slow's RD refinement of vectors and intra modes (x264's subme 8). The time
gap after the third round is still implementation efficiency, now spread
thin: a veryfast park_joy profile is a sixth trellis, a seventh pricing, a
tenth candidate coding, a twelfth the bitstream writer, a twelfth intra
analysis, and the rest search, motion compensation and ~8 % copying the
2 KB decision records between candidates; medium's is a third search
(two thirds of that sub-sample SATD of partitions), a fifth trellis and a
sixth pricing. No single piece is more than a few per cent from here: the
next steps are batching (a macroblock's sixteen transforms and quantisers
in one kernel, several search candidates per SAD/SATD call), decision
records referenced rather than copied, and a trellis over all nodes in
vector lanes.

### H.265 compression against x265

The same `--preset veryfast | medium | slow` selects the H.265 encoder's
tier (`Config::preset`, Medium by default; `src/encode/h265_tune.rs`). Under
`--codec h265` the flag also sets that codec's structure — three B pictures
in the scheduler's pyramid, SAO, references 1 / 2 / 4 — and the tier's
quantiser offsets: the configured QP is a P picture's, I pictures code 3
below it, the reference B 2 above and the others 4 above (only under
`Config::pyramid`, so a constant-QP stream without one stays constant).
`Config::legacy_decisions` gives the untiered decisions.

What every tier decides with (`h265_me`, `h265_intra`, `h265_search`,
`h265_rdoq`): SATD-domain costs at the square root of the SSD Lagrangian
(they had been priced at the SSD one, nine times too dear at QP 32: +51 %
→ +30 % on its own); a motion search priced in vector bits, seeded with the
parent unit's vector and reading every fractional phase from planes each
reference computes once through the decoder's kernels; each unit weighed
in SSD + λ·bits with and without its residual, and coded intra (2Nx2N, and
NxN at 8x8) where that is cheaper; rates from an estimating CABAC encoder
that starts each CTU from the contexts the picture has reached (every CTU
decided is run through the writer); temporal motion vector prediction (at
a constant quantiser: the SPS flag, a B slice's collocated picture its
future anchor, a P slice's its nearest past one); an inter CU's transform
tree split once into four quarter TUs where that is cheaper, tried in 8x8
CUs (4x4 luma TUs, the 4:2:0 chroma left whole); sign data hiding;
rate-distortion optimised quantisation of intra units and of inter units'
luma (inter chroma takes the plain quantiser); no SAO in pictures nothing
references. Slow adds RDOQ in those pictures too. Medium and veryfast
prune: a skip cheaper than 0.5 λ per sample (veryfast: every skip) is not
split, intra is coded only where its best mode's SATD cost beats the inter
search (slow: comes within 1.5x of it), angles are searched coarse to
fine, and pictures nothing references take the cheapest settings, RDOQ
off; veryfast's intra pictures do not split their transform.

Measured with `parity.py` (30 frames, QP 22/27/32/37 against x265 `--tune
psnr --crf 18/23/28/33`, x265 one thread; BD-rate, negative = fewer bits
than x265; CPU = summed child CPU seconds, rivet / x265, at `--jobs 4` on a
loaded machine, indicative to about ±0.05x):

| `--preset medium` vs x265 medium | BD PSNR-Y | BD PSNR-avg | CPU |
|---|---:|---:|---:|
| foreman CIF | −14.84 % | −14.49 % | 1.09x |
| mobile CIF | −9.14 % | −6.35 % | 1.03x |
| akiyo CIF | −21.79 % | −20.22 % | 1.05x |
| ducks_take_off 720p | −3.60 % | −2.69 % | 0.90x |
| park_joy 720p | −3.39 % | −2.10 % | 1.06x |
| **mean** | **−10.55 %** | **−9.17 %** | **0.98x** |

| `--preset veryfast` vs x265 veryfast | BD PSNR-Y | BD PSNR-avg | CPU |
|---|---:|---:|---:|
| foreman CIF | −13.23 % | −12.48 % | 1.15x |
| mobile CIF | −7.08 % | −4.27 % | 1.15x |
| akiyo CIF | −22.17 % | −20.60 % | 1.01x |
| ducks_take_off 720p | −10.76 % | −9.32 % | 1.02x |
| park_joy 720p | −6.19 % | −4.91 % | 1.17x |
| **mean** | **−11.88 %** | **−10.32 %** | **1.09x** |

| `--preset slow` vs x265 slow | BD PSNR-Y | BD PSNR-avg | CPU |
|---|---:|---:|---:|
| foreman CIF | −0.77 % | −1.90 % | 0.78x |
| mobile CIF | +2.82 % | +4.96 % | 0.75x |
| akiyo CIF | −13.81 % | −13.83 % | 0.97x |
| ducks_take_off 720p | +7.96 % | +8.40 % | 0.74x |
| park_joy 720p | +6.70 % | +7.43 % | 0.84x |
| **mean** | **+0.58 %** | **+1.01 %** | **0.79x** |

The CPU column of these tables is round 3's (2026-11-15), measured with
develop's binary run back to back at the same load: develop took 1.27x /
1.05x / 1.09x (veryfast / medium / slow) in the same session where the
user's own measurement of it was 1.22x / 1.05x / 1.13x, so the ratios
above are within about 0.05x of that machine's state.

Where it started (the same harness, `--bframes 3 --refs 4 --sao` at constant
QP): +46.7 % against medium at 2.31x the CPU, +65.7 % against slow, +40.5 %
against veryfast. The steps, mean BD-rate against medium: the SATD
Lagrangian +51.3 → +29.8 (CIF clips); per-kind quantiser offsets and the
pyramid +1.85; dropping residual by RD +0.35; the priced search −0.49;
intra by RD −3.46 (and NxN −4.40); sign hiding −6.06 at 1.62x; the medium
pruning −3.35 at 1.1x. The second round (2026-11-15): live contexts −4.17;
temporal MVP −5.31 (veryfast −5.09 → −6.97, slow +5.96 → +3.93); the
split inter transform tree, 8x8 to 32x32, −8.18 at 1.19x (veryfast −9.98,
slow +2.48; without the 8x8 CUs medium gained only −0.74), then 8x8 only
−7.53 at 1.05x (slow +2.53 at 1.10x, veryfast −9.11 at 1.21x); slow's
intra trials without a transform split and behind a 1.5x SATD gate,
+3.93 → +4.03 at 0.78x the CPU; no inverse transform for a block with no
levels, 0.97x, byte-identical.

The third round (2026-11-15) was speed first, all of it byte-identical
except where named, and then RDOQ at every tier with what the speed paid
for. CPU against the same session's develop binary, medium unless named:
decisions decided into recycled 6 KB decisions and pooled trial saves
instead of zeroed, returned and boxed ones (0.94x); RDOQ with an early
exit for a block nothing rounds up in, no 8 KB prefix array, last-position
prices by table per prefix and sign hiding from the known last position
(slow 0.85–0.90x); one merge-list derivation per unit where five were
made (0.97x); a reference's fractional phases built over the picture and
16 samples around it rather than the 80-sample padding (CIF 0.93–0.96x;
park_joy's streams change, its far-padding phases not having been exactly
the interpolated values, BD unchanged); the residual scratch at the block's
size and the transform tree priced once (0.97x); mvd bits by table
(0.94x); SAO rail buckets only within reach of a spellable offset (park_joy
0.96x). Then the tools: RDOQ on inter luma and intra units at every tier,
not on inter chroma (with it: medium −8.78 %, without −10.38 %; slow
+2.52 → +0.79 %), and not in pictures nothing references at medium and
veryfast (medium −10.28 % at 0.95x the CPU of −10.38 %; veryfast −9.11 →
−11.76 %, ducks +0.50 → −11.03 %); no SAO in pictures nothing references
(veryfast −12.06 %, medium −10.55 %, slow +0.58 %, each at 0.92–0.95x);
veryfast's intra pictures without a transform split (−11.88 % at 0.97x).

What is left, and why. Medium and veryfast now beat x265 on every clip;
medium's mean CPU is under x265's but its CIF clips are 1.03–1.09x, and
veryfast is 1.09x (1.15x on foreman, mobile and park_joy). Veryfast's
remaining CPU is the full RD decision at every quadtree node (residual,
RDOQ and pricing of each candidate, ~40 % of its time), SAO decisions in
the reference pictures (~8 %) and intra in inter pictures (~9 %); dropping
the 8x8 transform split there buys 6 % for 1.4 % BD (mobile −7.1 → −3.3 %),
quarter-sample refinement on a cross 1.3 % for 1.1 %, SAO in reference
pictures off 11 % for 2.4 % — none kept. Slow is +0.58 % at 0.79x: ducks
+8.0 %, park_joy +6.7 %, mobile +2.8 %. Its spare CPU found nothing to
buy: the split transform tried in 16x16 and 32x32 CUs (+0.57 / +0.48 % at
1.06 / 1.11x), intra transform splits in inter pictures (+0.57 %), a
second widening ring (+0.57 %), the losing merge/AMVP shape coded too and
the cheaper kept (+0.56 % at 1.2x), quantiser offsets I −4 / B +1, +3, +5
(+0.48 / +0.84 / +0.92 / +0.93 %). The 720p gap is inter coding of fine
moving texture (the second round's per-picture finding, P pictures 0.2–0.3
dB below x265 at matched bits) and is not a tool this encoder lacks among
those tried.

The lookahead quantiser propagation (`h265_cutree`, `--tree N`) is not the
720p gap, though x265 loses 21.9 % on ducks and 14.3 % on park_joy without
its own. x265's per-frame log (black box) shows why: with propagation on,
its frame quantisers on ducks are I ≈ P, reference B +2.2, other B +4.5 —
the structure rivet's fixed offsets (−3 / 0 / +2 / +4) already have —
while `--no-cutree` gives I 25 and P 36, a structure its rate factor
rather than propagation sets. Rivet's propagation on top of its offsets
doubles the frame-level part (medium, strength 2.0: −3.35 → +3.09 %);
mean-normalised to the spatial part alone it is neutral (−2.37 % at 2.0,
−3.11 % at 1.0); replacing the offsets with it (I 0, B 0 / +4) loses
(+0.62 % at 2.0, −2.02 % at 1.0). No tier turns it on.

Tried and dropped (medium unless named): an early skip (best merge coded
first, kept when nothing survives) −0.5 % BD for 0.99x; diamond sub-sample
patterns −1.2 to −2.0 % for 0.95–0.98x; search widening rings −0.2 % for
1.06x (kept at slow); limiting references to the parent's −0.9 % for 0.96x;
two references instead of three −0.07 % for 0.94x (kept); a flat GOP at its
best offsets +12.0 %, pyramids of 5 and 7 +6.3 / +4.7 % against +1.85 % at 3;
RDOQ at medium −2.2 % BD for 1.31x CPU (slow only; with live contexts and
TMVP −5.73 against −5.31 % at 1.30x, the CIF clips 1–2 % worse); RDOQ
pricing at live contexts at slow +4.32 % against +3.93 % at initial ones;
RDOQ lambda 0.7x / 1.3–1.4x worse at both tiers; the Lagrangian 0.5 / 0.65 /
0.8 / 1.25x −3.87 / −4.01 / −3.91 / −1.77 % against −3.35 % (before live
contexts, CPU +3–16 %); a 1.5x Lagrangian on non-reference B pictures
−2.06 %; the inter dead zone at a tenth / 0.22 of a step −2.36 / −3.75 %;
the intra one at a quarter −3.22 %; symmetric partitions at slow, with the
transform split, +3.82 % against +3.77 % at 2.1x the CPU; the transform
split on 16x16 and 32x32 CUs (see above); no split in pictures nothing
references −7.43 % against −8.18 % (all sizes) at 0.96x; quarter-sample refinement on a cross at
veryfast −5.48 % against −6.97 %; a conservative all-zero test before the
forward transform (a bound on the coefficients from the residual's SAD)
never fired on the corpus; temporal MVP under a bitrate (the rate
controller's three-GOP windows went outside [0.85, 1.18] on the cut clip
with a lookahead); RDOQ applied only to each CTU's final inter units after deciding
without it (medium −6.26 % against −7.53 % without RDOQ at all: the
decisions must see the quantiser); RDOQ's chroma Lagrangian divided by the
chroma step weight the decisions give chroma SSD (medium −8.44 % against
−8.78 %); RDOQ off for intra chroma too (medium −9.87 %, slow +1.30 %);
stopping the quadtree at a residual-free unit below 1 λ per sample at
veryfast (−10.46 % for 0.95x).

## Threading and SIMD

Both decoders are threaded two ways, on one FIFO worker pool sized to the
machine (`H26X_THREADS`, default every core, up to 32):

- **Frame threading.** Pictures decode concurrently; a picture that references
  another waits only for the rows it needs (motion compensation waits on the
  reference's *filtered* row progress, temporal motion prediction on its
  *decoded* row progress), and the deblocking / SAO of a picture trail its
  decode row by row so the next picture can start before this one ends. Up to
  `H26X_INFLIGHT` pictures are in flight (default `threads.clamp(2, 16)`).
- **Intra-picture parallelism (H.265).** Wavefront rows (WPP), tiles and slice
  segments each run as their own task; a row waits on the CTB above-right, and
  the next row is spawned as soon as this one is two CTBs in. Dependent slice
  segments resume their predecessor's CABAC state.

**Threads on WebAssembly.** `wasm32-unknown-unknown` has neither
`std::thread` nor a clock, even when it is built with atomics and shared
memory and threads do exist (each a Web Worker running another instance of
the module on the same memory, which only JavaScript can start). An embedder
that can start them installs a [`platform::Platform`](src/platform.rs) — a
function that runs a thread body, and a millisecond clock — with
`h26x::platform::install`, and from then on every pool asks it for its
workers and the decoders read its clock. Native builds never install one and
are unchanged; without one, a decoder on wasm runs with one thread.
[wasm-h26x](https://github.com/safewords/wasm-h26x) is that embedder.

The pixel kernels — interpolation, weighting, inverse transforms, residual add,
SAO for H.265; the sixteen quarter-sample positions, chroma bilinear, averaging
and weighting for H.264 — sit behind a function table filled at run time from
what the CPU has. On x86-64 that is a ladder — **SSE2**, **SSSE3**,
**SSE4.1**, **AVX**, **AVX2** — climbed one rung at a time, each rung
replacing only the kernels it improves on the one below, so a CPU ends up with
the best available version of every kernel. SSE2 is baseline on x86-64, so the
bottom rung always applies: no x86-64 machine runs the scalar kernels, which
are the executable specification the others are tested against and not a
fallback. AArch64 has **NEON**, likewise baseline, with the ARMv8.2-A **dot
product** extension above it where present. **AVX-512** (F + BW + VL) sits
above AVX2 as a supplement rather than a rung: it replaces the handful of
H.265 kernels whose block shape genuinely fits 512-bit lanes and leaves the
rest of the AVX2 table alone.

Measured on one core against the scalar reference, **SSE2 alone is worth
2.5–3.2x and is nearly all of the total.** What the rungs above it add depends
on the codec, and the split is large: everything from SSSE3 to AVX-512
together is worth about **5% on H.264** but **13–17% on H.265**. Block width
is the reason. H.264's blocks are at most sixteen samples wide, so a 256-bit
vector spans a row at best and pays cross-lane permutes to undo per-lane
packing that 128-bit code never does; H.265 has 32- and 64-wide blocks, and
its x86 kernels already put two rows in a vector. So the individual rungs —
SSSE3 (`pmaddubsw` for the six-tap and bilinear filters), SSE4.1
(`pblendvb`, `pminsd` / `pmaxsd`), AVX (VEX's three-operand encoding; AVX adds
no 256-bit *integer* operation, that is AVX2) — are worth low single digits
each on a modern out-of-order core, and should be worth more on the processors
that actually need them, which have far less slack to hide the extra
instructions the rung below spends. On AArch64, `sdot` is worth having for the
one kernel whose shape it fits, HEVC's eight-tap horizontal luma filter, and
measurably not worth it for H.264's six-tap, whose taps LLVM already folds
into `umlal` pairs.

`H26X_NO_SIMD=1` forces the scalar kernels and
`H26X_MAX_SIMD=avx512 | avx2 | avx | sse41 | ssse3 | sse2 | neon | none` caps
the ladder one rung at a time, which is how one machine checks that every rung
decodes to the same bytes — `tools/verify.sh` does exactly that on every run
and fails if any two rungs disagree. Every SIMD kernel is also checked
bit-exact against the scalar reference by the crate's tests: on x86-64
in CI, on aarch64 by hand (see "NEON on ARM hardware"). `h26xdec --rung` (or [`dsp::Cpu::rung`]) prints which rung
was selected, which is worth knowing before quoting a number from this
machine or any other. `H26X_PROF=1` prints where the time went.

### NEON on ARM hardware

CI runs on x86-64 Linux only, so the NEON (aarch64) code paths are not tested
there. They are verified by hand on ARM hardware (an aarch64 Linux machine,
or Apple silicon) after a change to them and before a release:

```sh
cargo test --release
H26X_NO_SIMD=1 cargo test --release
# on a CPU with the dot-product extension (Apple M-series, Graviton 3, ...):
H26X_REQUIRE_DOTPROD=1 cargo test --release
```

The first run checks every NEON kernel bit-exact against the scalar
reference; the second decodes and encodes everything again on the scalar
code; the third fails, rather than skips, if the `dotprod` rung is missing.

## Performance

The tables below were measured (2026-09) against libavcodec, on the same
clips, with both decoders materialising every frame. They are kept as a
record; the tooling no longer runs any FFmpeg program, so
`tools/benchmark.py` now regenerates them with the scalar rung as the
baseline (a `vs scalar` column) instead of a libavcodec row. Every frame is
materialised: this decoder packs each picture and drops it. Neither writes to disk. **Cost** is CPU seconds
(user+kernel) for the whole process tree, best of five; **throughput** is
frames per wall second, which is the question a multi-threaded run is actually
asking. Regenerate with `tools/benchmark.py`.

Rows are the instruction-set rungs, because "how fast is it" has no answer
without saying which instructions it was allowed to use — and the rung is
chosen at run time, so the table doubles as a map of what different hardware
gets. `H26X_MAX_SIMD` caps the ladder, which is how one machine produces every
row; the rung it selects on its own is in bold, and is what a user on that
hardware actually runs.

One line under each table is generated and says "Widest rung against
libavcodec". For the two H.264 tables that widest rung is byte-identical to
the row beneath it, since no H.264 kernel is AVX-512 — so it names a
configuration that is not distinct for this codec, and those two rows should
be read as a control pair rather than as two rungs.

The single-threaded rows are pinned to one quiet core, both decoders alike, so
that column survives a machine that is doing other things. The all-threads
rows cannot be — pinning them would measure something other than what they
claim — so they are the rows to distrust on a busy box, and the header says
how busy it was.

**AMD Ryzen 9 9950X 16-Core Processor**, 32 hardware threads, selecting **AVX-512**. Machine 17% busy before the run. Single-threaded rows are pinned to core 29 (both decoders). Best of 5. Cost is CPU seconds, throughput is frames per wall second.

#### `cabac3.264` — H.264, 1280x720, 723 frames

| instructions | 1 thread: CPU s | vs libav | fps | all threads: CPU s | fps |
|---|---:|---:|---:|---:|---:|
| **AVX-512** | 2.266 | 1.49x | 319 | 3.766 | 2169 |
| AVX2 | 2.219 | 1.46x | 326 | 3.766 | 2221 |
| AVX (VEX-128) | 2.266 | 1.49x | 319 | 3.781 | 2206 |
| SSE4.1 | 2.266 | 1.49x | 319 | 3.875 | 2123 |
| SSSE3 | 2.281 | 1.51x | 317 | 3.703 | 2153 |
| SSE2 | 2.297 | 1.52x | 315 | 3.969 | 2163 |
| scalar | 5.469 | 3.61x | 132 | 7.734 | 1329 |
| libavcodec | 1.516 | 1.00x | 477 | 2.625 | 2119 |

Widest rung against libavcodec: **1.49x** its single-threaded CPU time, **1.43x** with every thread; SIMD is worth **2.4x** over the scalar reference.

#### `cavlc3.264` — H.264, 1280x720, 723 frames

| instructions | 1 thread: CPU s | vs libav | fps | all threads: CPU s | fps |
|---|---:|---:|---:|---:|---:|
| **AVX-512** | 1.672 | 1.55x | 432 | 3.031 | 3116 |
| AVX2 | 1.656 | 1.54x | 437 | 3.078 | 2997 |
| AVX (VEX-128) | 1.734 | 1.61x | 417 | 3.344 | 2890 |
| SSE4.1 | 1.734 | 1.61x | 417 | 3.500 | 2856 |
| SSSE3 | 1.766 | 1.64x | 409 | 3.594 | 2831 |
| SSE2 | 1.703 | 1.58x | 425 | 3.062 | 2985 |
| scalar | 4.906 | 4.55x | 147 | 7.406 | 1550 |
| libavcodec | 1.078 | 1.00x | 671 | 2.406 | 3063 |

Widest rung against libavcodec: **1.55x** its single-threaded CPU time, **1.26x** with every thread; SIMD is worth **2.9x** over the scalar reference.

#### `hevc6.265` — H.265, 1280x720, 1446 frames

| instructions | 1 thread: CPU s | vs libav | fps | all threads: CPU s | fps |
|---|---:|---:|---:|---:|---:|
| **AVX-512** | 3.328 | 1.12x | 434 | 4.219 | 1288 |
| AVX2 | 3.406 | 1.15x | 425 | 4.391 | 1277 |
| AVX (VEX-128) | 3.656 | 1.23x | 395 | 4.266 | 1241 |
| SSE4.1 | 3.688 | 1.24x | 392 | 4.562 | 1255 |
| SSSE3 | 3.734 | 1.26x | 387 | 4.578 | 1228 |
| SSE2 | 3.844 | 1.29x | 376 | 4.500 | 1220 |
| scalar | 8.578 | 2.89x | 169 | 9.375 | 693 |
| libavcodec | 2.969 | 1.00x | 487 | 4.141 | 1387 |

Widest rung against libavcodec: **1.12x** its single-threaded CPU time, **1.02x** with every thread; SIMD is worth **2.6x** over the scalar reference.

#### `wpp10.265` — H.265, 1280x720, 1200 frames

| instructions | 1 thread: CPU s | vs libav | fps | all threads: CPU s | fps |
|---|---:|---:|---:|---:|---:|
| **AVX-512** | 1.938 | 1.02x | 619 | 4.656 | 1963 |
| AVX2 | 1.984 | 1.04x | 605 | 4.062 | 1946 |
| AVX (VEX-128) | 2.172 | 1.14x | 553 | 4.609 | 1799 |
| SSE4.1 | 2.203 | 1.16x | 545 | 4.359 | 1913 |
| SSSE3 | 2.188 | 1.15x | 549 | 4.859 | 1749 |
| SSE2 | 2.281 | 1.20x | 526 | 4.297 | 1792 |
| scalar | 5.125 | 2.69x | 234 | 7.984 | 1067 |
| libavcodec | 1.906 | 1.00x | 630 | 2.656 | 1774 |

Widest rung against libavcodec: **1.02x** its single-threaded CPU time, **1.75x** with every thread; SIMD is worth **2.6x** over the scalar reference.

### Reading it

**Two things in the table say how much precision it can carry, and both are
worth reading before any of the numbers.**

The first is inside each H.264 table. No H.264 kernel is AVX-512 — the tier
carries H.265 kernels only — so the top two rows of those two tables run
*byte-identical code*, and every difference between them is measurement noise.
There is no single number for it, because it depends on the column: 2.1% and
1.0% in single-threaded CPU seconds, 0.0% and 1.6% in all-thread CPU seconds,
and **2.4% and 4.0% in all-thread frames per second**. The threaded columns
are the noisiest, and they are the ones the multi-threaded claims below are
made in, so read those against 4% and not against 2%.

The second is between runs. This table replaced one taken earlier on the same
machine, and no H.265 code changed in between — yet `wpp10.265` moved from
0.96x to 1.02x against libavcodec and `hevc6.265` from 1.11x to 1.12x. The
absolute costs of *both* decoders drift with whatever else the machine has
been doing, and the ratio inherits it. So **treat the comparison column as
good to about ±5% between sessions**, and do not read a change of a few per
cent in these figures as a change in the code. A few per cent in the code is
measurable, but only by holding everything else fixed — one binary, two code
paths, switched by an environment variable — which is what
`tools/verify.sh --baseline` and the per-change measurements in the commit log
do, and what a table like this cannot.

**With that said: single-threaded, H.265 is at parity with libavcodec** (1.02x
and 1.12x) and H.264 costs about half as much again — 1.46-1.49x on a CABAC
stream, 1.54-1.55x on a CAVLC one. Those are ranges rather than figures
because the two H.264 rows they come from are the same code: the bolded row is
whichever rung this machine selects, not whichever sample came out faster, and
on both clips it happens to be the slower one. Quoting it alone would inflate
the gap by the width of the control. The gap itself is not in the pixel
kernels; it is entropy decoding and per-macroblock bookkeeping, which is where
the remaining work is.

**With every thread, two of the four comparisons say anything.** Ahead on WPP
(1963 against 1774 fps, +10.7%) and behind on the H.265 clip without
wavefronts (1288 against 1387, -7.1%). The other two do not clear the floor:
CAVLC (3116 against 3063) is +1.7% and CABAC (2169 against 2119) is +2.4%,
against a byte-identical control spread of 4.0% and 2.4% in that same column.
Frame threading is the whole of it for H.264; H.265 adds wavefront rows, tiles
and slice segments inside a picture.

**Nearly all of the SIMD win is on the bottom rung, and on H.264 there is now
almost nothing above it.** Scalar to SSE2 is worth 2.4–2.9x. Everything above
SSE2 put together adds **1–2% on H.264** — at or under the floor the table
itself reports — against **15–18% on H.265**. Block width is why: H.264's
blocks are at most sixteen samples wide, so a 256-bit vector spans a row at
best and pays cross-lane permutes to undo per-lane packing that 128-bit code
never does, while H.265 has 32- and 64-wide blocks and its x86 kernels already
put two rows in a vector. The H.264 figure has *fallen* as this decoder got
faster: removing redundant work from the interpolation filters took out the
part that the wider rungs were helping with, leaving a profile even more
dominated by entropy decoding than before. On x86-64 nothing ever falls back
to scalar, since SSE2 is baseline, so the practical reading is that an old
x86-64 machine gives up almost nothing on H.264 and about a sixth on H.265.

**AVX-512 is a supplement, not a rung.** It installs over a finished AVX2
table and replaces only the H.265 kernels whose shape suits 512-bit lanes: the
vertical 14-bit stage of a diagonal interpolation, the wide luma filters at 32
samples and up, and the 32-point inverse transform, whose row *is* a 512-bit
vector. Everything it does not carry keeps the AVX2 version — H.265 chroma
interpolation was written for it, measured worse, and was left behind, as were
four H.264 kernels.

Its end-to-end contribution is the one figure in this section with no control
behind it. The byte-identical pair exists only in the H.264 tables, because
AVX-512 genuinely does carry H.265 kernels — so the H.265 tables have no two
rows running the same code, and the H.264 floor cannot be borrowed across to
them when that floor is itself anything from 0.0% to 4.0% depending on the
column. Per-kernel it is 1.6-2.3x on the shapes it carries; what that is worth
inside a decode was measured separately, against its own control, and is
recorded in the commit that added the tier rather than inferred from here.

Numbers move with the clip. These are 720p Big Buck Bunny encodes — H.264 High
profile from x264 at CRF 20 with B-pyramid and the 8x8 transform, H.265 Main
from x265 with and without WPP — repeated to a length the clock can measure
(process CPU time comes in ~15.6 ms steps, so a fifth-of-a-second clip cannot
be compared with itself, let alone with something else).

### Encoder speed

The encoders share the decoders' interpolation, inverse transforms and
loop filters, and since 2026-08-27 have SIMD tiers of their own for the
kernels an encode profile actually spends its time in: the distortion
metrics (SAD, SATD, SSD), the H.265 forward transforms and quantiser, and
(since 2026-09-18) the H.264 forward transforms and quantisers, each on the same three architectures as the decode kernels — SSE2 to
AVX2, NEON (bit-exact, checked by hand on ARM hardware) and wasm `simd128`
(bit-exact and round-tripped inside the module by `tools/wasm.sh`) — and
`tools/verify_enc_ladder.sh` holds every `H26X_MAX_SIMD` rung of an encode
to the scalar reference's bytes. The CABAC encoder's
per-bin cost is a table now rather than a `log2` per bin, and RDOQ skips a
block whose last coefficient is three or more. Since 2026-11-15 the H.264
slice data is written by `CabacWriter` (bytes out with a carry into one held
byte, the same bits as the standard's `PutBit`, held to it by the module's
round trip), candidates are priced by `CabacEstimator` (a bin is one fused
(state, bin) table lookup; the writers are generic over `BinEnc` so each
compiles for it), the trellis's tables are sized to the block, and the
search's quarter-sample averages run in fixed-width rows: together 0.82x /
0.84x / 0.86x of the time at veryfast / medium / slow, every stream
byte-identical. The record, with the
profiles that chose these and the controls behind every number, is
`docs/encode_speed.md`.

Measured as `tools/ab_enc.py` measures — one machine, interleaved runs of
the two builds pinned to the quietest core, median of nine paired
CPU-second ratios — 640x360 `testsrc2`, 90 frames, QP 26, the tree before
this work (5cb468c) against after:

| | H.264 intra | H.264 IP | H.264 IPB | H.264 t8x8+subparts | H.264 CAVLC IP | H.265 intra | H.265 IP | H.265 IPB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CPU time, after / before | 0.689 | 0.490 | 0.420 | 0.446 | 0.477 | 0.653 | 0.527 | 0.485 |
| paired range (9 rounds) | 0.667–0.694 | 0.469–0.521 | 0.406–0.441 | 0.445–0.452 | 0.455–0.488 | 0.646–0.660 | 0.513–0.549 | 0.481–0.492 |

Two same-binary controls in the same session read 1.000 (0.931–1.037)
and 1.000 (0.991–1.009); a row's difference from 1 is real only where it
clears that. The streams are byte for byte the ones the scalar kernels
produce — `tools/identity_encode.sh` encodes every gate cell both ways and
compares (280 identical, 0 moved) — so this is speed and nothing else.
`H26X_ENC_NO_SIMD=1` (or a table name: `distortion`, `hevc_enc`) keeps the
encode-side tables scalar for a measurement; `H26X_MAX_SIMD` caps the
ladder as it does for the decoders. Neither encoder is threaded yet:
`--threads` is accepted and unused. Against x264 at matched tiers see
"H.264 compression against x264" above: 2.2x to 3.3x x264's single-thread
time, the gap in rivet's scalar candidate pricing, residual coding and
search control flow rather than in how much it searches.

## Provenance and licensing

This is not a translation of libavcodec's C, which is LGPL and could not be
carried under this crate's license. The decoders were written from the ITU-T
Recommendations (H.264 08/2024, H.265 V11 01/2026) — the decoding process is
fully specified there — using libavcodec's *architecture* as the model:
parameter-set tables, a slice-driven decode loop over a decoded-picture buffer,
entropy decoding into per-block syntax, then prediction, inverse transform,
reconstruction and in-loop filters, with the pixel kernels behind a
runtime-dispatched DSP layer. Numeric tables (VLC codes, CABAC initialisation
values, scan orders, filter taps) are the standard's tables.

**Patents.** The AVC and HEVC patent pools (Via LA, Access Advance, Velos)
license *products and services* — encoders, decoders, and content shipped to
end users — not the act of writing or publishing an implementation; several
pools additionally waive royalties for application software distributed at no
charge. Nothing here is a licence to any patent, and anyone shipping a product
built on this crate is responsible for their own patent position, exactly as
with x264, x265, OpenH264 or FFmpeg. Source availability was never the
licensed act; distribution of a decoding product is.

## Using it

Feed whole NAL units (Annex-B framed, parameter sets included) in decode order;
pictures come back in output order as [`Picture`](src/picture.rs) — cropped
planar samples (8-bit as bytes, 9–16-bit as little-endian 16-bit words; with
unequal luma / chroma depths every plane takes the wider word, as HM writes
them) with the chroma format, both bit depths and picture order count of the
stream:

```rust
let mut dec = h26x::hevc::HevcDecoder::new();   // or h26x::h264::H264Decoder
dec.push_annexb(&bytes)?;                        // whole NAL units, Annex-B framed
while let Some(pic) = dec.next_picture() { /* h26x::Picture, output order */ }
dec.flush()?;                                    // drain the reorder buffer
```

`examples/h26xdec.rs` decodes a raw `.264` / `.265` file and prints one line per
output frame with the MD5 of the packed planar picture — the same layout
the reference decoders' YUV output holds — or writes raw YUV. Debug environment variables:
`H26X_NO_DEBLOCK`, `H26X_NO_SAO`, `H26X_TRACE=<mbaddr>|all` (a macroblock's
parsed layer and motion), `H26X_TRACE_IPM=1` (every syntax element with its
bit / CABAC position — lines up with the JM reference decoder's trace) and
`H26X_TRACE_DPB=1` (picture starts, marking, lists) (H.264), `H26X_TRACE_PS`
(parsed SPS/PPS), `H26X_TRACE_CU`, `H26X_TRACE_PU=x,y`,
`H26X_TRACE_TB=c,x,y`, `H26X_VERIFY_HASH=1` (check every decoded-picture-hash
SEI, count a warning per mismatch) (H.265).

## License

Open Encoding Attribution License v1.0 — a source-available (not OSI open-source)
license, royalty-free, with a commercial-attribution requirement. See
[LICENSE.md](LICENSE.md) and [NOTICE](NOTICE).
