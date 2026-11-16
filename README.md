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
  on SATD plus rate, each ring's points scored four at a time
  (`satd_avg_x4`). P pictures search up to 3 (veryfast 2, slow 5) references, B pictures nothing
  references two in list 0 (medium and slow), and a
  P partition below 16x16 picks its own among those whose 16x16 cost came
  within 1.25x (slow 1.5x) of the best (mixed references). A B pair is
  refined jointly — each list's vector a quarter sample around with the
  other held, scored as the average they make, the held list's
  prediction made once a ring (16x16: one round in veryfast, two
  otherwise, and only from within 1.05x of the better list; slow refines
  its partitions' pairs too, from within 1.1x), and
  list 1's search also starts from list 0's vector mirrored.
- **Decision**: shapes, B directions and intra modes are estimated at SATD
  plus rate. A P macroblock whose 16x16 estimate already trails
  Intra_16x16's (medium 1.0x, slow 1.1x) searches no partitions, and one
  whose best inter estimate trails it by 1.1x (slow 1.2x) codes no inter
  candidate (medium also stops searching references at 1.2x); then skip, the best one to three inter shapes at both transform
  sizes and the best intra type are then coded and compared on
  `SSD + lambda * R`, with `R` priced by the picture's own CABAC writer from
  its live context states (`CabacEncoder::estimating`) — no table of what
  syntax "usually" costs (`CabacEstimator`, the same writers compiled for
  pricing). Early skip when the skip prediction quantises to nothing and
  its error is small. Veryfast (since the seventh round) prices only I
  pictures: a P or B macroblock is decided on the estimates and only the
  winner is coded, its transform size from the residual's 4x4 and 8x8
  Hadamard SATDs, with skip priced against it where skip's estimate is no
  worse, intra analysed within 1.2x of the best inter estimate, and
  Intra_4x4 analysed in I pictures only.
- **Quantisation**: the chosen macroblock is re-quantised by a trellis over
  each block's level choices, priced from the live contexts
  (`h264_trellis.rs`). Medium and slow quantise every candidate the
  comparison codes by the trellis, so their decisions see the levels they
  will write; veryfast trellises only the winner. A block the plain
  inter quantiser (dead zone at 1/6) leaves all zero is not trellised.
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
| veryfast | foreman CIF | +2.50 % | +2.67 % | 1.33x |
|  | mobile CIF | -9.15 % | -9.78 % | 1.48x |
|  | akiyo CIF | -6.56 % | -5.97 % | 1.95x |
|  | ducks_take_off 720p | -12.17 % | -13.36 % | 1.34x |
|  | park_joy 720p | -3.76 % | -3.61 % | 1.32x |
|  | **mean** | **-5.83 %** | **-6.01 %** | **1.35x** |
| medium | foreman CIF | +1.51 % | +1.90 % | 1.12x |
|  | mobile CIF | +2.22 % | +2.10 % | 1.14x |
|  | akiyo CIF | -4.85 % | -4.59 % | 1.36x |
|  | ducks_take_off 720p | -0.56 % | -3.09 % | 1.08x |
|  | park_joy 720p | +3.49 % | +3.71 % | 1.04x |
|  | **mean** | **+0.36 %** | **+0.01 %** | **1.07x** |
| slow | foreman CIF | +2.04 % | +2.41 % | 0.96x |
|  | mobile CIF | +2.12 % | +1.75 % | 0.78x |
|  | akiyo CIF | -1.91 % | -1.48 % | 1.24x |
|  | ducks_take_off 720p | +2.22 % | -1.30 % | 0.76x |
|  | park_joy 720p | +2.73 % | +2.78 % | 0.74x |
|  | **mean** | **+1.44 %** | **+0.83 %** | **0.76x** |

(After round 10B, 315c0bc, 2026-11-16, rivet at --jobs 4 against the
cached x264 times; the frontier runs below and under "Round 10A" and
"Round 10B" timed x264 again beside rivet; at equal BD-rate rivet now
needs 1.04x / 1.15x / 1.14x x264's time. The ninth round's table read
1.44x / 1.19x / 0.82x at -5.83 / +0.33 / +1.44 %; its notes follow. Cycles at
QP 27 over the five clips, minimum of three runs, `/usr/bin/time -l`:
5.50G / 14.09G / 23.53G before (47bb0bb), 5.17G / 11.84G / 18.73G
after, -6.0 / -15.9 / -20.4 %; instructions 23.6G / 62.3G / 108.3G to
22.9G / 53.5G / 86.3G, -3.1 / -14.0 / -20.3 %. x264 at CRF 23 on the
same clips in the same session: 3.35G / 9.76G / 20.84G cycles, 12.1G /
34.4G / 74.7G instructions — rivet is now 1.54x / 1.21x / 0.90x its
cycles and 1.89x / 1.56x / 1.16x its instructions. The eighth round's
table, for the record: -5.84 / +0.23 / +1.25 % at 1.48x / 1.36x /
1.01x.)

**The efficiency frontier.** Comparing each preset with x264's
same-named one hides where rivet stands, so since the eighth round all
three are placed on x264's whole ladder: every x264 preset from
ultrafast to veryslow on the same clips and settings, BD-rate Y against
x264 medium (mean of the five clips' BD-rates) against CPU summed over
clips and rate points, relative to x264 medium's. Every encode, x264's
and rivet's (the tree before this round and after), ran under
`/usr/bin/time -l` in one shuffled batch, four at a time, minimum of
three runs each (`frontier.py`, a companion to the parity script;
2026-11-16):

| encoder | BD-rate Y vs x264 medium | CPU vs x264 medium |
|---|---:|---:|
| x264 ultrafast | +228.64 % | 0.127x |
| x264 superfast | +47.00 % | 0.200x |
| x264 veryfast | +20.30 % | 0.348x |
| x264 faster | +7.29 % | 0.645x |
| x264 fast | +1.96 % | 0.878x |
| x264 medium | 0 | 1x |
| x264 slow | -3.58 % | 2.100x |
| x264 slower | -6.05 % | 3.832x |
| x264 veryslow | -6.65 % | 5.385x |

Every x264 preset is on its own frontier (none is beaten on both axes by
another). Rivet's, against that curve (x264 interpolated piecewise
linearly in BD-rate against log CPU between its neighbouring presets):

| rivet | BD-rate Y | CPU | x264's BD-rate at this CPU | x264's CPU at this BD-rate | rivet's time at equal BD-rate |
|---|---:|---:|---:|---:|---:|
| veryfast, 47bb0bb | +13.07 % | 0.568x | +9.97 % | 0.490x | 1.16x |
| veryfast, now | +13.09 % | 0.548x | +10.73 % | 0.490x | **1.12x** |
| medium, 47bb0bb | +0.23 % | 1.422x | -1.70 % | 0.985x | 1.44x |
| medium, now | +0.33 % | 1.227x | -0.99 % | 0.978x | **1.25x** |
| slow, 47bb0bb | -2.39 % | 2.376x | -4.09 % | 1.641x | 1.45x |
| slow, now | -2.21 % | 1.945x | -3.21 % | 1.580x | **1.23x** |

(The eighth round's run of the same table read 1.15x / 1.45x / 1.46x for
the tree before this round: the measurement repeats to about 0.01.)
All three are still above x264's frontier, now by 1.12x / 1.25x (1.254
unrounded) / 1.23x the time x264 needs for the same BD-rate — at each
one's time x264 is 2.4 / 1.3 / 1.0 points better. By time, veryfast
sits between x264 veryfast and faster, medium between x264 medium and
slow, slow just short of x264 slow; the names hold and the presets were
not re-cut. The frontier says what a trade is worth where each preset
sits: x264's own slope is about 0.2 % BD-rate per 1 % of time between
veryfast and faster, 0.05 % between medium and slow, 0.04 % past slow.
Every decision change this round bought time at 0.02 % per 1 % or less
(the slow ones at 0.02 to 0.03 %); trades at the slope itself were left
alone, since they move a preset along x264's curve rather than toward it.

What the ninth round did. The eighth round's profile was flat and its
decision trades all sat near x264's slope, so this round looked for two
other things: cycles the machine throws away (a top-down count of
park_joy at medium has 19 % of the issue slots discarded on mispredicted
paths), and work whose outcome is already known from estimates the
analysis has anyway. Byte-identical first (cycles and instructions at
QP 27, five clips, minimum of three runs, each against the tree before
it):

| change | veryfast | medium | slow |
|---|---:|---:|---:|
| trellis: the walk back through the chosen path writes every position's level (zero included) and counts it without a branch on the level — the positions that keep one are as good as random, and each mispredicted | -2.9 % cycles (+0.5 % ins) | -3.7 % (+0.5 %) | -3.2 % (+0.6 %) |
| CABAC estimator: the significance map a position at a time, the last flag priced by a mask rather than a branch (`BinCore::decision_if`, `BinEnc::PRICES`) | -0.9 % (+0.2 %) | -0.9 % (+0.7 %) | -1.4 % (+0.7 %) |
| ... levels of one and two without a branch on which | -0.4 % (+0.1 %) | -0.5 % (+0.5 %) | -0.4 % (+0.6 %) |
| sub-sample refinement: the ring's points scored four at a time against one load of the source (`H264EncDsp::satd_avg_x4`, NEON; a plane sample is the pair of one row twice) | +0.6 % (-1.8 % ins) | -0.5 % (-4.1 %) | -0.1 % (-4.1 %) |
| B pair refinement: the held list's prediction made once a ring, its moving list's points four at a time (`satd_bi_x4`) | -0.9 % (-1.7 %) | -2.5 % (-3.5 %) | -4.5 % (-7.1 %) |
| the comparison's candidate order in an array, not a vector allocated per macroblock | noise | noise | noise |

Then decisions, each found by counting what the pruned work would have
won (BD-rate Y against x264 at the same preset, cycles at QP 27 against
the step before):

| change | veryfast | medium | slow |
|---|---:|---:|---:|
| P macroblocks: a 16x16 estimate no better than Intra_16x16's (1.1x in slow) searches no partitions, and a best inter estimate 1.1x behind it (1.2x) codes no inter candidate (`part_lead`, `rd_lead`) | | +0.23 → +0.34 %, -6.5 % | +1.25 → +1.35 %, -3 to -5 % |
| B 16x16 pairs refined only from within 1.05x of the better list (`bi_ratio`) | -5.84 → -5.83 %, -0.5 % | → +0.34 %, -0.9 % | → +1.30 %, -0.5 % |
| P: a third reference not searched when the 16x16 estimate so far is 1.2x Intra_16x16's (`ref_lead`) | | → +0.33 %, -1.2 % | |
| slow: further inter shapes coded only within 1.1x of the best estimate (was 1.25x), partition pairs refined only from within 1.1x (was 1.2x) | | | → +1.42 %, -4.8 % |
| slow: one quarter-sample round for partitions (was two) | | | → +1.44 %, -2.9 % |

The P measurement that chose the first row (park_joy, medium, QP 27): of
the P macroblocks whose 16x16 estimate was 1.2x Intra_16x16's or more,
11,500 went intra, none to a 16x16 and 20 to a partition shape; of those
whose best inter estimate was 1.2x Intra_16x16's or more, none went
inter, and 3 of 1,940 between 1.1x and 1.2x. On park_joy two P macroblocks in
three end up intra (the P pictures are four apart), and each had paid for
three references' searches, partitions, and two shapes coded at both
transform sizes. The B pair: of the pairs starting 1.1x or more behind
the better single list, one in 12,300 (park_joy) ended as the 16x16
choice after refinement and none as the macroblock's.

Tried and dropped in the ninth round. Decisions (BD-rate Y against x264
at the same preset, cycles against the kept tree): medium with fewer or
more B pictures, `--bframes` 1 / 2 / 4 / 5 / 7, +9.22 / +4.94 / +4.18 /
+5.49 / +8.37 % (three is a sharp optimum under the pyramid's offsets,
park_joy included, though two P macroblocks in three are intra there);
the intra lead at 0.9 / 1.0 (partitions / coding) +0.52 % for 1.8 % more,
at 0.9 / 1.1 +0.42 %, and coding at 1.0 with the third reference at 1.1
or alone, +0.42 % for -1.3 to -1.7 % (at the slope); slow's lead at 1.0 /
1.1, +0.24 % for 8 % (good, but past the +1.5 % ceiling with the rest),
its third-reference lead at 1.2, +0.10 % for 1.2 %, one half-sample round
for partitions +0.06 % for 2.2 %, partition range 4 +0.12 % for nothing.
The transform size from the 8x8 / 4x4 Hadamard ratio, measured rather
than tried: the ratio sits between 0.95 and 1.1 for most candidates and
both sizes win often there; only past 1.2 does the 4x4 always win, too
rare to pay. Speed (byte-identical, no gain or a loss): an early exit
from a sub-sample candidate's SATD once it passes the best cost (85 % of
park_joy's candidates lose, but by so little that at most 4 % of the
SATD work could be skipped); the ones-only trellis without a branch on
zero positions (+1.2 % instructions, -0.3 % cycles, noise); the general
trellis's zero step folded into its ones step (+2.4 % instructions,
+0.9 % cycles); the CABAC writer's significance map branch-free like the
estimator's (no change: the writer codes each block once).

Where the time goes now. A top-down count (Instruments' CPU Counters,
park_joy, medium, before this round) put 52 % of issue slots to use, 24 %
waiting on the back end, 19 % discarded after mispredictions, 5 % on
instruction delivery; x264 medium on the same clip, 40 / 25 / 24 / 12 %.
After the round (medium, park_joy and ducks_take_off, sampled): SATD
kernels and the refinement around them about a fifth, the trellis a
sixth (its per-position branch on the rounded level is the largest
single source of mispredictions left; making it branch-free costs more
work than it saves), the estimator 7 %, the final writer 5 %, intra
analysis in inter pictures a tenth. What is left between medium and
x264's curve is about 0.3 points of BD-rate or a fifth of its time;
slow's is a fifth of its time; veryfast's an eighth, where the final
CABAC writer alone is a sixth of the time and changes no decision.

#### Round 10A: the final CABAC writer and the trellis

(2026-11-16. The tenth round was split in two; this part had the final
CABAC writer and the trellis, the other the motion search, the SATD
kernels and the estimator. Every change here is byte-identical: the
BD-rates are the ninth round's to the digit, at all three presets.)

Cycles and instructions at QP 27 over the five clips, minimum of three
runs, `/usr/bin/time -l`, each change against the tree before it:

| change | veryfast | medium | slow |
|---|---:|---:|---:|
| CABAC writer: `low` sixty-four bits wide, its bits out thirty-two at a time into a byte buffer of the writer's own (a carry adds one to the bytes already there), rather than a byte at a time with a pending byte and a run of `0xff`s | -3.4 % cycles (-1.3 % ins) | -1.7 % (-0.6 %) | -0.9 % (-0.3 %) |
| trellis: the first pass in NEON — exact levels four at a time, the scan gathered by byte lookups, the last position and the largest level from a mask and a reduction, no array zeroed first | -1.3 % (-1.4 %) | -1.0 % (-1.4 %) | -1.1 % (-1.5 %) |
| trellis: the walk back through the chosen path by a shift and a mask per position (the source lanes, times eight, a byte each in one word), not an indexed load | -0.8 % (+0.2 %) | -1.1 % (+0.2 %) | -1.3 % (+0.2 %) |
| **all three** (5.27G / 11.93G / 18.85G before) | **-6.0 %** (-2.4 %) | **-3.7 %** (-1.8 %) | **-3.3 %** (-1.6 %) |

On x264's ladder (`frontier.py`, x264 timed again in the same shuffled
batch, four at a time, minimum of three runs; BD-rate against x264
medium):

| rivet | BD-rate Y | CPU | x264's CPU at this BD-rate | rivet's time at equal BD-rate |
|---|---:|---:|---:|---:|
| veryfast, be1409c | +13.09 % | 0.543x | 0.492x | 1.10x |
| veryfast, now | +13.09 % | 0.518x | 0.492x | **1.05x** |
| medium, be1409c | +0.33 % | 1.226x | 0.978x | 1.25x |
| medium, now | +0.33 % | 1.177x | 0.978x | **1.20x** |
| slow, be1409c | -2.21 % | 1.949x | 1.583x | 1.23x |
| slow, now | -2.21 % | 1.877x | 1.583x | **1.19x** |

(This run read 1.10x for veryfast before the round where the ninth
round's read 1.12x; the others repeat to 0.01.) H.265 is untouched: its
writer is the standard-shaped `CabacEncoder`, and 18 streams (three CIF
clips, three presets, QP 22 and 32) are byte-identical before and after;
H.265 medium parity is still -14.45 %. The decoder is untouched: 56
streams (the decoder fixtures and `tests/data`) decode to the same
picture MD5s.

The writer, measured on its own. A replay of park_joy's residual blocks
as veryfast writes them (738,000 blocks, 19.1 million bins: 30 frames
at QP 27; the blocks dumped from an encode and coded again by the real
writer, cycles from `/usr/bin/time -l` with the replay's set-up
subtracted) took 17.1 cycles and 50 instructions a bin before this
round and 13.6 cycles and 43 instructions after. Almost all of that is
the branch that sent bytes out: crossed every eight or so bins at a
point the data decides, it was mispredicted nearly every time, and the
path behind it (the pending byte, the run of `0xff`s, a call into the
bit writer per byte) cost more than the bins between. A top-down count
of the replay (Instruments' CPU Counters) had half its issue slots
discarded after mispredictions before the change.

How the counts were taken. Branch mispredictions can be sampled, not
only counted: `xctrace record --template 'CPU Counters'
--recording-options` with a JSON whose `configurationType` is `manual`,
`sampleByTime` false and `pmiEventAliasOrMnemonic`
`BRANCH_MISPRED_NONSPEC` gives a profile whose samples land where
mispredicted branches resolve (the kernel throttles the interrupts, so
the counts are relative, not absolute). On park_joy at medium the
trellis had 19 % of the samples, the estimator's residual pricing 8 %,
the sub-sample SATD 12 %, the final writer 5 %.

Tried and dropped (byte-identical unless said otherwise):

- The writer coding a residual block from a list: the block's bins
  built first without a branch on its values (every position's
  significance and last flags written and the count moved past the ones
  not coded; a level's prefix of ones written twelve ahead and the count
  moved by its length; each sign folded into the bin before it as one
  more doubling of `low`), then coded in one straight run. The run itself
  was made cheaper again: the range kept in the top of a word, a bin's
  new range one of four candidates (`range - rLPS` or `rLPS` for each
  quarter of the range, each one add of a constant loaded from the state
  and the bin) picked by two selects on the range's own bits, then a
  count of leading zeros and a shift — six steps from one range to the
  next where the writer takes nine. In the replay that reached 12.8
  cycles a bin (the run 7.7 of them, the list 4); in the encoder, -0.4 %
  cycles at veryfast against the change above alone, +2.7 %
  instructions: noise, and dropped. The same six-step bin in the
  writer's ordinary per-bin path: +0.9 % at veryfast. The replay keeps
  the writer's tables and contexts in the first-level cache; the encoder,
  which runs a motion search between two macroblocks' bins, does not.
- Carrying a context's state in a register from one bin to the next
  when they share the context, selected against a load from the array:
  slower, because the select waits on both, which ties every bin to the
  one before through the state tables (in the replay 14.1 against 13.2
  cycles a listed bin without it).
- The trellis's ones-only path without a branch on the rounded level,
  every position's prices taken before the walk and the costs
  re-associated so that one add per lane leads to a position's
  candidates (not byte-identical): +38 % in a replay of the trellis's
  ones-only blocks. Its general path with a rounded one priced as a two
  with the two made infinite, which leaves one branch per position
  (zero or not) instead of three: +2 % (4x4) and +8 % (8x8). Why the
  branch on zero stays: a level of one is three dependent adds and a
  select from the position before (about thirteen cycles), a zero two
  adds (six); a branch-free step costs every position thirteen, more
  than the mispredictions it removes. What did pay was everything around
  that walk: the first pass was, in a replay of park_joy medium's
  trellised blocks, two fifths of a ones-only block's time, and the walk
  back a chain of dependent loads.

Where the time goes now (park_joy, sampled): at veryfast the trellis
13 % and the final writer 12 %; at medium the trellis 17 % (every coded
candidate's blocks) and the final writer 6 %.

#### Round 10B: the motion search and its kernels

(2026-11-16. The other half of the tenth round: the motion search, the
SAD / SATD kernels around it and the candidate pricing. One change moves
medium's decisions; everything else is byte-identical, and veryfast and
slow keep the ninth round's BD-rates to the digit.)

Cycles and instructions at QP 27 over the five clips, minimum of three
interleaved runs, `/usr/bin/time -l`, each change against the tree
before it (measured before round 10A was merged; the total is against
round 10A's tree):

| change | veryfast | medium | slow |
|---|---:|---:|---:|
| the three half-sample planes of a reference in one allocation, each a quarter, a half and three quarters of a 16 KiB span past the integer plane — allocated apart, all four planes of every reference were page-aligned, so one sample of each fell in the same L1D set (park_joy medium: L1D load misses 166M to 90M) | -1.3 % cycles (-0.0 % ins) | -1.4 % (-0.0 %) | -2.3 % (-0.1 %) |
| medium: a 16x16 search at a further reference (P past the first, each B list past its first) whose whole-sample cost, SAD plus rate with its index's bits, is 1.5x the best an earlier one reached is not refined to sub-samples (`refine_ratio`; `Search::run` split into `int_search` and `refine`) | | +0.33 → +0.36 %, -2.4 % (-2.4 %) | |
| the planes, the block's offset and the rate table read out of the search once rather than reloaded through two or three dependent loads after every kernel call; the rate table a fixed array, its index clamped to it | +0.2 % (-0.4 %) | -0.1 % (-1.2 %) | -0.7 % (-1.6 %) |
| half-sample rings scored by a four-way SATD with no average (`H264EncDsp::satd_x4`, NEON): every half-sample point is a plane sample, which the average-SATD loaded twice and averaged with itself | -0.8 % (-1.0 %) | -1.0 % (-1.4 %) | -1.5 % (-2.0 %) |
| **all four, on round 10A's tree** (4.87G / 11.31G / 18.03G before) | **-1.6 %** (-1.4 %) | **-5.3 %** (-5.1 %) | **-4.3 %** (-4.0 %) |

The gate's measurement (park_joy, medium, QP 27, before it): of P
searches at a further reference 1.5x or more behind at whole samples,
4 of 17,500 would have won the 16x16 after refinement; of B ones, 25 of
18,300. Mobile is the clip it costs (+0.24 %; the others moved within
the corpus's noise). Branch mispredictions (park_joy, PMI-sampled
counts): 70.4M before, 69.9M after at medium, 104.9M / 104.5M at slow —
this round's changes are not where they are: the trellis has 38 % of
them, the final writer 18 %, the estimator 15 % and the whole search 7 %
(before round 10A).

On x264's ladder (`frontier.py`, x264 timed again in the same shuffled
batch, four at a time, minimum of three runs; BD-rate Y against x264
medium):

| rivet | BD-rate Y | CPU | x264's CPU at this BD-rate | rivet's time at equal BD-rate |
|---|---:|---:|---:|---:|
| veryfast, round 10A (f0ccc51) | +13.09 % | 0.518x | 0.489x | 1.06x |
| veryfast, now | +13.09 % | 0.510x | 0.489x | **1.04x** |
| medium, round 10A | +0.33 % | 1.182x | 0.978x | 1.21x |
| medium, now | +0.36 % | 1.123x | 0.976x | **1.15x** |
| slow, round 10A | -2.21 % | 1.884x | 1.582x | 1.19x |
| slow, now | -2.21 % | 1.804x | 1.582x | **1.14x** |

(Round 10A's own run of its tree read 1.05x / 1.20x / 1.19x: the
measurement repeats to about 0.01.)

What limited the ninth round's `satd_avg_x4`. Not memory: it had half
of all L1D load misses (81M of 166M, park_joy medium), and halving them
with the skewed planes left its share of the time where it was (11.9 %).
Its calls that run (1.13M at 16x16, 0.48M at 16x8 and 8x16, 0.73M at
8x8) cost, at a hot-cache microbenchmark's rates (75 cycles a 16x16
candidate, 99 a call at 8x8), about 500M cycles; in the encode it takes
540M. Out-of-order execution hides the misses. What binds it is SIMD
issue: a 4x4 tile pair is about 32 vector operations (the differences,
two butterfly stages, eight `trn`s, the folded last stage, `sadalp`),
about one per sample, and the four-way kernel's shared source rows saved
loads and instructions, not vector operations — 89 cycles a 16x16
candidate alone, 75 four at a time, against a floor near 64. Taking the
redundant average out of plane-sample candidates (four `urhadd`s and
four loads a candidate stripe) took it to 68.

Tried and dropped. Kernels (hot microbenchmark, 16x16, cycles a
candidate): the horizontal transform as `sdot` dot products against
the Hadamard rows on bias-shifted bytes, the source's transform folded
into the accumulators, 86 against 75 (sixteen `sdot`s, sixteen
accumulator copies and eight narrowings a stripe are more instructions
than the transposes they replace); the four candidates unrolled with
sixteen-bit accumulators, 84 (LLVM's loop prefetching and register
pressure); a branch-free refinement (masks for the points a ring
scores, selects for its winner, a ring that did not move followed by an
empty one rather than a branch out), identical output, -0.1 % cycles
for +0.3 % instructions. Decisions (BD-rate Y against x264 at the same
preset, cycles against the kept tree): `refine_ratio` at 1.3, medium
+0.27 % for -3.0 %; at 2.0, -0.01 % for -1.2 %; slow at 1.5, +0.17 % for
-2.6 % and at 2.0 +0.07 % for -1.4 % (x264's slope there is 0.04 to
0.05 % per 1 %); veryfast at 1.3, +0.12 % for -0.5 %; the same gate on a
P partition's references and a B partition's two lists at 1.5, medium
+0.16 % for -0.8 %, slow +0.34 % for -1.7 %; medium with one
quarter-sample round, +0.10 % for -1.5 %, or one half-sample round,
+0.15 % for -1.4 %. The estimator was left as it is: its bins are
already two table loads each from the live state, and its
mispredictions (the significance and level loops' exits and the test
for levels above two) are in the residual walk it shares with the final
writer that round 10A was changing.

Where the time goes now (medium, ducks_take_off and park_joy, sampled,
before round 10A): the four-way SATD kernels about a ninth, the trellis
a fifth, the estimator 8 %, the integer search with its SAD kernel 7 %,
the refinement's own code 4 %.

The eighth round, for the record (its frontier: 1.15x / 1.45x / 1.46x).

What the eighth round did (BD-rate Y against x264 at the same preset,
instructions at QP 27, each step against the one before):

| change | veryfast | medium | slow |
|---|---:|---:|---:|
| CABAC writer: a context's four LPS sub-ranges one word, loaded from the state alone, the range picking a byte by shift — the table load is off the range's dependency chain (byte-identical) | -1.9 % cycles | | |
| trellis: blocks the plain inter quantiser (dead zone 1/6) leaves all zero are not trellised (2 % of them had kept a level) | -2.80 → -2.66 %, -1.4 % | +0.39 → +0.23 %, -3.0 % | +1.38 → +1.25 %, -3.1 % |
| trellis: the callers' level arrays arrive zeroed, the trellis's own `bzero` per 8x8 gone (byte-identical) | -0.7 % | -1.3 % | -1.4 % |
| veryfast: two P references | → -3.94 % (alone), +3.1 % | | |
| veryfast: the trellis in B pictures nothing references | → -3.85 % (alone), +2.3 % | | |
| veryfast: both | → -5.12 %, +5.5 % | | |
| veryfast: intra analysed within 1.2x of the best inter estimate (was 1.0x) | → -5.84 %, +1.0 % | | |

Of the trellis's calls (park_joy, QP 27), 49 % (veryfast) / 44 %
(medium) find every exact level under a half and return at once, a
further 16 % / 19 % are the dead-zone blocks now skipped, and of the
rest — blocks with a level the dead zone keeps — the trellis changes at
least one level in 86 % (veryfast) and 80 % (medium): there is no large
class of blocks the dead zone already decides, so a trellis run only on
"ambiguous" blocks has nothing to gain. The trellis's inner loop is
already table-driven (each context's bin costs priced once per block
from the live states, no CABAC inside it).

Tried and dropped in the eighth round (BD-rate Y, instructions, against
the kept tree at the time). Veryfast: three P references, a further
-0.48 % for +3.4 % (0.14 % per 1 %, under x264's 0.2 % there); a second
list-0 reference in non-reference B pictures, -0.26 % for +3.8 %; the
quarter-sample square instead of the diamond, -0.78 % for +4 % (at the
slope); two half-sample rounds, -0.20 % for +2.2 %; two B pair rounds,
-0.23 % for +3.3 %; the B pair square instead of the diamond, -0.12 %
for +4.4 %; intra within 1.5x, a further -0.29 % for +2.2 %; B intra
within 1.2x, -0.04 % for +1.1 %; Intra_4x4 in inter pictures, +0.26 %
for +4.8 %; partitions below 16x16 (`--subparts`), -3.5 % for +52 %.
Medium (each about x264's 0.05 % per 1 % there, so frontier-neutral,
and each spends BD-rate the +0.5 % ceiling leaves little of): the B pair
diamond +0.18 % for -3.4 %; mixed references within 1.1x +0.19 % for
-3.0 %; the B partition gate at 0.95 +0.13 % for -2.9 %; B intra within
0.8x +0.02 % for -0.4 %; two P references +0.67 % for -3.8 %. The dead
zone gate at 0.7 (+0.25 %) and 0.9 (+0.31 %) of a step instead of 5/6,
at 1.0 (+0.83 %), and luma-only (+0.65 %): 5/6 is both principled and
the best measured. (The BD-rate on these 30-frame clips moves by about
0.2 % between near-identical configurations; trades smaller than that
are not resolved.)

Where the time goes now (veryfast, park_joy, QP 27, sampled): the trellis
and the final CABAC residual writer about a seventh each (the writer about
15 cycles a bin before this round; it is latency-bound on the range's
renormalisation chain, and the stream is what it is), SATD and fused
average-SATD of the sub-sample search an eighth, the lookahead 5 %.
Medium: the trellis a fifth (every coded candidate's blocks), the
estimator's residual pricing 9 %, sub-sample SATD 16 %, the integer
search and its control flow 12 %, the final writer 5 %. Not tried this
round: the lookahead on a quarter-size picture (it is already
half-resolution and 5 % of veryfast; mode decisions there would move
decisions, not the frontier); batched four-way SATD for the sub-sample
rings (the kernels are 16 % of medium; the most a shared source load can
save is a fraction of that).

The seventh round, for the record: -2.80 / +0.39 / +1.38 % at 1.44x /
1.44x / 1.06x. (Measured 2026-11-15, rivet at --jobs 4 against the cached x264
times. The sixth round's tree, 3ec8b66, timed in the same session: 1.93x
/ 1.50x / 1.12x at -4.68 / +0.31 / +1.40 %. Cycles at QP 27 over the five
clips, minimum of three runs, `/usr/bin/time -l`: 7.23G / 15.31G / 25.97G
before, 5.19G / 14.62G / 24.44G after, -28 / -4.5 / -5.9 %; instructions
30.4G / 66.4G / 116.8G to 22.8G / 64.7G / 112.6G, -25 / -2.6 / -3.6 %.
x264 at CRF 23 on the same clips in the same session: 3.36G / 9.76G /
20.91G cycles, 12.1G / 34.4G / 74.6G instructions — rivet is now 1.55x /
1.50x / 1.17x its cycles and 1.9x / 1.9x / 1.5x its instructions.)

What the seventh round did. Veryfast had BD-rate to spend (it was 4.7 %
ahead of x264 veryfast, a far lighter configuration) and now decides P
and B macroblocks the way that preset's name suggests; medium and slow
had none, and took only what cost nothing or next to it (BD-rate Y
against x264, instructions at QP 27, each step against the one before):

| change | BD-rate Y | instructions |
|---|---:|---:|
| veryfast: P and B macroblocks decided on their estimates, only the winner coded and (in reference pictures) trellised; skip no longer competes on its prediction's error; the winner's transform size from its 4x4 and 8x8 Hadamard SATDs (`sa8d_16x16`) | -4.68 → -0.84 % | -17 % |
| ... intra candidates estimated without coding chroma or Intra_16x16, the winner alone coded again | byte-identical | -1.2 % |
| ... I pictures still decided by pricing | → -1.73 % | +1.0 % |
| ... the trellis in non-reference B pictures too | → -3.07 % | +3.9 % |
| ... the B pair refined on the diamond (`bi_diamond`) | → -2.88 % | -4.4 % |
| ... skip priced against the coded winner where skip's estimate is no worse (`skip_rd`) | → -3.73 % | +1.6 % |
| ... Intra_4x4 in I pictures only (on the estimates it won far more often than pricing let it) | → -3.98 % | -4.4 % |
| ... early skip's error ceiling 0.06 → 0.10 | → -3.95 % | -0.6 % |
| ... the trellis in non-reference B pictures off again | → -2.80 % | -3.9 % |
| veryfast in all: -4.68 → -2.80 % | +1.88 % | -25 % (cycles -28 %) |
| medium: B pictures analyse intra within 1.0x of the best inter estimate (was 1.5x) | +0.31 → +0.39 % | -2.4 % |
| slow: the same within 1.25x (was 2.0x) | +1.40 → +1.38 % | -3.4 % |
| SA8D in vector-shaped lanes; the trellis's exact levels built without a zeroed array | byte-identical | -2 % veryfast, -0.2 % medium |

(The trellis in non-reference B pictures is worth 1.2 % for 4 % of the
instructions here, about the 0.3 % per 1 % every other cut left costs,
and spending that is what the headroom is for.) The decision the
estimate path makes is what the tier had before it priced: the +35 % measured in the first round was skip competing on
the SATD of its prediction against the coded candidates' SATDs, which
leave out the residual those candidates then code — most macroblocks a
residual would have mended went to skip. Skip now comes only from early
skip, a coded 16x16 at the skip vector with nothing left, or the priced
comparison.

Tried and dropped in the seventh round (BD-rate Y, instructions at QP
27). Veryfast, on top of the kept configuration, all near 0.3 % per 1 %:
no trellis on the winner either, +5.7 % for -15 %; Intra_4x4 / 8x8
candidates estimated from the source rather than the reconstruction,
+0.7 % for -2 %; fewer Intra_4x4 / 8x8 modes (vertical, horizontal, DC
and the predicted mode, then only the angles beside the winner), +0.6 /
+1.0 % for -2.4 / -3.6 %; one B pair refinement round dropped, +1.2 %
for -3.5 % (and no fewer cycles); intra to 0.9x, +0.6 % for -0.7 %;
the half-sample ring as a diamond, +1.1 % for -2.5 %; no quarter
samples, +5.2 %; the skip comparison off, +0.9 % for -1.6 %; priced at
1.3x, -0.2 % for +2.4 %; a penalty on Intra_4x4's estimate, nothing;
dead-zone rounding instead of the trellis for blocks with a level of
five / three / two or more, +1.0 / +1.6 / +2.3 % for -1.3 / -2.5 / -4.2 %; `--tree 0`, +2.3 % for -5 %; bounding the intra
analysis by the inter estimate (byte-identical, 0.04 %). Medium: the
transform size by SA8D instead of coding both, +1.35 % for -8 %; the 8x8
trial skipped only when SA8D is 1.1x / 1.2x the 4x4 SATD, +0.06 / +0.00 %
for no measurable saving; partitions skipped when the 16x16 cost is under
0.5 / 1.0 of early skip's SAD threshold, +0.03 / +0.64 % for -0.2 /
-1.5 %; later references' sub-sample refinement only within 1.15x / 1.3x
of the best whole-sample cost, +0.89 / +0.38 % for -4.4 / -2.4 % (slow
+1.34 / +0.73 % for -9 / -6 %); mixed references within 1.1x, +0.27 % for
-2.7 %; the B pair diamond, +0.22 % for -3.3 % (slow +0.25 % for -6 %,
past its ceiling); B intra within 0.8x, +0.10 % for -2.7 %.

Where the time goes now (veryfast, park_joy and ducks_take_off, sampled):
the trellis a sixth to a fifth (it is worth 5.7 % here), the CABAC
writer of the final stream a sixth (about 17 cycles a bin; the stream is
the same size as before), the sub-sample search a tenth, intra analysis
in inter pictures a tenth, the lookahead 4 %. Medium is unchanged in
shape: the trellis in every priced candidate a fifth, the search a
quarter, the estimator an eighth.

(Sixth round, 2026-11-15, rivet at --jobs 4 against the cached x264
times; every stream on all three presets is byte-identical to the fifth
round's, so the BD-rates are unchanged and the time is all speed. Cycles
at QP 27 over the five clips, minimum of three runs, `/usr/bin/time -l`:
8.52G / 17.11G / 29.13G before (3c2ed48), 7.24G / 15.10G / 25.76G after,
-15 / -12 / -12 %; instructions 38.9G / 81.4G / 141.7G to 30.4G / 66.4G /
116.9G, -22 / -18 / -17 %. x264 at CRF 23 on the same clips in the same
session: 3.38G / 9.84G / 21.84G cycles, 12.1G / 34.4G / 74.7G
instructions.)

What the sixth round did, every change byte-identical (instructions at
QP 27, veryfast / medium / slow, each against the tree before it):

| change | veryfast | medium | slow |
|---|---:|---:|---:|
| Intra_16x16 found once per macroblock (the estimate's mode handed to the analysis), its rows written as fixed-width slices | -2.7 % | -1.6 % | -1.3 % |
| Intra_4x4 / 8x8: the neighbours gathered (and filtered) once per block, each mode predicted into its own buffer (`Edge4` / `Edge8`, `pred_4x4` / `pred_8x8`) | -2.6 % | -2.2 % | -2.1 % |
| trellis: an all-zero block found from its largest exact level before the scan; Qstep without `exp2`; the general path's eight lanes as two NEON vectors, each destination's candidates a tree of first minima (`Trellis::simd`) | -1.5 % | -2.9 % | -3.1 % |
| NEON SATD: tile pairs folded with `sadalp`, `sabd` for the difference magnitudes, sixteen-wide rows in one load (shared with H.265) | -2.4 % | -3.8 % | -3.3 % |
| NEON prediction copy and two-/four-wide stores without `memmove`; chroma rows loaded once (shared with the decoder) | -1.6 % | -1.0 % | -0.9 % |
| difference and forward transform in one NEON pass (`sub_fdct4` / `sub_fdct8`); 4x4 dequantisation's shift chosen once per block | -4.3 % | -2.4 % | -2.4 % |
| CABAC residual writer: the significant positions as one mask (NEON table lookup), `i16` blocks straight in; the walk's coded_block_flag increments a plane at a time | -5.5 % | -3.5 % | -3.2 % |
| candidates' luma predicted from the search's half-sample planes instead of the six-tap kernels | -2.3 % | -1.3 % | -1.1 % |
| smaller: 8x8 sub-scan counts as masked sums, no memsets in the trellis's scan, the B search's pair average in the fused average-SATD | -2 % | -1.6 % | -2 % |

Tried and dropped in the sixth round: the ones-only trellis path in NEON
(slower than its scalar form — the per-position chain, not the
arithmetic, is its cost); a per-macroblock memo of trellis inputs (9 % of
medium's and 19 % of slow's trellis calls repeat one within the
macroblock, not enough to pay for the lookup); pricing chroma once per
candidate (chroma is 11-14 % of the priced coefficients, about 1 % of the
time); caching the lookahead's propagation fractions and landing blocks
(0.1 %). Decision trades, BD-rate Y against x264 (all over the 0.05 %
per 1 % of time allowed): medium partitions from their candidates without
an integer descent, +0.54 % for -2.6 % instructions; veryfast early-skip
error ceiling 0.10, +0.07 % for -0.6 %; veryfast intra ratio 0.9, +0.87 %
for -2.3 %; veryfast's 8x8 transform tried only when the 4x4 one coded
two (three) quadrants, +0.34 (+0.62) % for -0.6 (-1.3) %.

Where the time still goes (medium, park_joy, sampled): the trellis about
a fifth (latency-bound: a ones-only block is ~100 ns, a general 4x4 ~190
ns, a general 8x8 ~570 ns), the motion search a fifth (sub-sample SATD
at the kernel's floor, ~22 SATDs per 16x16 search, and the refinement's
control flow), the CABAC estimator's and writer's bins about a seventh
(one context per bin, at the coder's serial floor), and intra analysis
in inter pictures about a tenth. Rivet still executes 2.5x / 1.9x / 1.6x
x264's instructions; the profile is flat, with nothing left over a few
per cent that can be had without changing a decision, and the decisions
that would cut most (veryfast's rate-distortion pricing, medium's trellis
in every candidate) cost far more than the allowed rate.

(Fifth round, 2026-11-15, rivet at --jobs 4 against the cached x264
times; veryfast and slow streams are byte-identical to the fourth
round's. The fourth round's tree, 3b2cc06, timed in the same session:
2.32x / 2.04x / 1.33x. Cycles at QP 27 over the five clips,
`/usr/bin/time -l`: 8.76G / 20.74G / 30.52G before, 8.53G / 17.17G /
29.50G after, -3 % / -17 % / -3 %; instructions 40.0G / 100.5G / 150.5G
to 39.0G / 81.4G / 141.7G. x264 at CRF 23 on the same clips: 3.40G /
9.85G / 20.99G cycles, 12.1G / 34.4G / 74.7G instructions.)

What the fifth round did:

| change | veryfast | medium | slow |
|---|---:|---:|---:|
| trellis: a block whose rounded levels are all zero or one (61 % of the trellised blocks in a park_joy medium encode) in four lanes, not eight — no path reaches a larger level's nodes (byte-identical) | -0.5 % ins | -1.5 % cycles | -1.7 % cycles |
| integer search: start candidates, hexagon / diamond steps and the square refinement scored four at a time (`H264EncDsp::sad_x4`, NEON) then taken in the old order (byte-identical) | -1 % ins | -2 % ins | -2 % ins |
| a coded candidate whose squared error alone reaches the best cost is not priced; intra candidates against the caller's bound (byte-identical) | | -0.5 % | |
| medium: a second inter shape only within 1.03x of the best estimate (was 1.1), B partitions only within 0.98x (was 1.02) | | +0.41 % BD, -13 % cycles | |

Where the time goes, measured by running each piece twice and counting
the cycles that adds (the fourth round's tree, QP 27, five clips): the
CABAC estimator's pricing of coded candidates 9 % / 7 % / 7 %
(veryfast / medium / slow), the motion search (integer descent and
sub-sample refinement, not the B pair refinement) 7 % / 24 % / 25 %, the
trellis 6 % / 16 % / 17 %. Timed per phase of a park_joy macroblock
(medium): search 35 %, coding and pricing the inter candidates 42 %,
intra 15 %. A search averages 18 SADs and 18 SATDs, half of them on 8x8
and smaller partitions; a trellised block averages 15 scan positions, 7
of them nonzero, and 44 % of the calls find nothing to quantise.

Tried and dropped in the fifth round (BD-rate Y against x264, cycles at
QP 27): medium with the trellis only on the winner, +1.17 % for -14 %;
one inter shape, +0.97 % for -17 %; intra to 1.2x, +0.57 % for -5 %;
early skip's error ceiling 0.05 / 0.15, +1.18 / +1.26 % for -4 / -6 %;
intra candidates analysed without the trellis and the winner
re-quantised, 1 % fewer instructions and no fewer cycles; intra types
chosen on their estimates with only the winner priced, +0.77 % for
-1 %; partitions' search range 2 on top of the kept pair, +0.13 % for
-1.5 %. Slow at a second-shape ratio of 1.1 / 1.05, +0.16 / +0.26 % for
-6 / -11 %, and with the B gate at 1.0 / 0.98, +0.39 / +0.48 % for
-6 / -11 % — good rates, but past slow's +1.5 % ceiling from +1.40 %.
Veryfast: early skip's error ceiling 0.12 with a descent range of 8,
+0.12 % for -2.3 % (within the rate, inside the noise, not taken); the
trellis in non-reference B pictures, -0.95 % for +2.3 %.

The fourth round, for the record (its table, 3b2cc06: 2.37x / 2.07x /
1.33x at -4.68 / -0.10 / +1.40 %):

| change | veryfast | medium | slow |
|---|---:|---:|---:|
| trellis: the eight reachable path states as vector lanes, each position a fixed gather (byte-identical; 219 -> 126 ns a block on blocks dumped from a park_joy medium encode) | -5 % cycles | -11 % cycles | -11 % cycles |
| B pictures nothing references: a second list-0 reference | | -0.37 % BD, +3 % cycles | -0.36 % BD, +1 % cycles |

Multiple B references are general in the syntax (the slice header's
override for both lists, `ref_idx_lX` in both writers for every explicit B
shape, list construction in the decoder's default order, one loop-filter
identity per picture since a picture can sit in both lists); the analysis
searches each list's 16x16 at every active index and binds the
macroblock's explicit partitions to the cheapest, with 3 bits of bias past
index 0, while direct predicts at its derived indices. What is configured
is narrower, because the rest measured worse (medium, BD-rate Y against
x264): both lists in every B picture +0.86 % (list 1's second entry is
mostly a picture list 0 already holds); list 0 in every B picture +0.09 %,
+0.02 % with the bias (a reference B picture's further indices cost the
pictures predicting from it their colocated zero motion); list 0 in
non-reference B pictures -0.10 % (kept); three references there +0.25 %;
the bias at 0 / 8 bits +0.09 / +0.03 % (in every B). Veryfast with two:
-4.97 % for +3 % time, not taken (0.1 % per 1 %).

Tried and dropped in the fourth round: intra candidates in P and B
pictures quantised plainly, the trellis only on an intra winner, medium
+0.31 % for -4 % cycles; the trellis's sums reassociated so each position
adds to the path cost once (not byte-identical), -4 % trellis time;
NEON for the trellis's lane additions and selections (LLVM leaves the
eight-lane arithmetic scalar either way), no faster than the gathers
alone. Implicit B weighting still changes nothing as configured: the only
pairs at unequal distances are the second list-0 reference's.

The third round, for the record:

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
multiple P references, and a second list-0 reference in non-reference B
pictures. x264 uses more B references than that (its log, foreman /
mobile at medium: list 0 index 1 in 9 / 24 % of B partitions, list 1
index 1 in 7 / 15 %), but here the extra indices lose where they cost a
reference B picture's colocated motion or duplicate list 0; per-partition
reference choice (rivet binds the macroblock to one per list) and the
implicit weighting between unequal distances are what is left of that.
Also missing: slow's RD refinement of vectors and intra modes (x264's
subme 8). Measured against x264's whole ladder after round 10B,
rivet needs 1.04x / 1.15x / 1.14x the time x264 needs for the same
BD-rate (1.12x / 1.25x / 1.23x after the ninth round, 1.16x / 1.44x /
1.45x before it, 1.30x / 1.54x / 1.55x before the eighth); veryfast is 5.8 % ahead of x264 veryfast. Rivet runs 1.9x /
1.6x / 1.2x x264's instructions at a higher IPC (1.5x / 1.2x / 0.9x its
cycles at QP 27). Veryfast's cuts still cost about 0.3 % per
1 % of time, more than x264's 0.2 % between veryfast and faster, so
none of them brings it nearer the frontier; what would is speed that
changes no stream. The fifth round's profile is flat — no
piece over a quarter — so what is left is many kernels each a few per
cent (the trellis's eight-lane state machine compiled scalar, SATD of
every sub-sample candidate, the estimator re-pricing a whole macroblock
per candidate, decision records copied by value) and, in veryfast, the
rate-distortion decision x264 veryfast does not make (it decides on
SATD at subme 2 without the trellis, and is 4.7 % behind for it). After the
fourth round a medium park_joy profile is a quarter trellis (4.2 million
blocks in 30 pictures, at about 125 ns each — the eight-lane gathers are
compiled scalar, and the remaining cost is the per-position dependency
chain and the call count of every coded candidate's 4x4 and chroma
blocks), a tenth SATD, a twelfth the estimator's residual pricing, and
about a third search (hexagon, sub-sample SATD of partitions), with
memmove and memset of the decision records under 5 %. The next steps are
fewer trellis calls per macroblock (or a cheaper first pass deciding
which blocks need one), batching (a macroblock's sixteen transforms and
quantisers in one kernel, several search candidates per SAD/SATD call),
and decision records referenced rather than copied.

### H.265 compression against x265

The same `--preset veryfast | medium | slow` selects the H.265 encoder's
tier (`Config::preset`, Medium by default; `src/encode/h265_tune.rs`). Under
`--codec h265` the flag also sets that codec's structure — three B pictures
in the scheduler's pyramid, SAO, references 1 / 2 / 4 (a B picture's lists
one each, at slow two each), symmetric partitions at slow — and the tier's
quantiser offsets: the configured QP is a P picture's, the reference B
codes 2 above it and the others 4 above, and an I picture between 6 below
and level with it as the lookahead propagation over the next sixteen
pictures says (below; only under `Config::pyramid`, so a constant-QP
stream without one stays constant).
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
CUs (4x4 luma TUs, the 4:2:0 chroma left whole); sign data hiding, the
level moved weighed in distortion and roughly in bins; rate-distortion
optimised quantisation of every unit, luma and chroma, which weighs each
block against coding nothing and searches its last position from the last
coefficient of 0.7 of a step; no SAO in pictures nothing references.
Slow adds RDOQ in those pictures too, iterative bi-prediction refinement, a
Lagrangian at 0.7 of the conventional one, two references per list in B
pictures, and 2NxN / Nx2N partitions searched from the 2Nx2N unit's motion
where that unit is not a merge, with a skip cheaper than 0.5 λ per sample
ending the recursion. Medium and veryfast
prune: a skip cheaper than 0.5 λ per sample (veryfast: every skip) is not
split, intra is coded only where its best mode's SATD cost beats the inter
search (slow: comes within 1.5x of it), angles are searched coarse to
fine, pictures nothing references take the cheapest settings, RDOQ
off, and the 8x8 transform split is tried only where the whole TU kept
four luma levels; veryfast's intra pictures do not split their
transform, a unit above 8x8 whose search costs more than 2.5 SATD
Lagrangians a sample (2.0) is split without being coded whole, three of
the five merge candidates are scored, RDOQ's last position is searched
from 0.85 of a step and SAO weighs two edge classes of four.

Measured with `parity.py` (30 frames, QP 22/27/32/37 against x265 `--tune
psnr --crf 18/23/28/33`, x265 one thread; BD-rate, negative = fewer bits
than x265). CPU is rivet's one-thread CPU over x265's on the same twenty
encodes per clip, the two (and develop's binary) run side by side at the
same moment so they meet the same load (`tcmp.py`, round 5): steadier than
the `--jobs 4` column `parity.py` prints, which read 0.88x / 0.90x / 0.90x
for the same binaries.

| `--preset medium` vs x265 medium | BD PSNR-Y | BD PSNR-avg | CPU |
|---|---:|---:|---:|
| foreman CIF | −17.50 % | −15.49 % | 0.97x |
| mobile CIF | −11.94 % | −8.17 % | 0.94x |
| akiyo CIF | −25.37 % | −24.46 % | 0.95x |
| ducks_take_off 720p | −9.83 % | −5.19 % | 0.84x |
| park_joy 720p | −7.59 % | −4.28 % | 1.00x |
| **mean** | **−14.45 %** | **−11.52 %** | **0.92x** |

| `--preset veryfast` vs x265 veryfast | BD PSNR-Y | BD PSNR-avg | CPU |
|---|---:|---:|---:|
| foreman CIF | −13.79 % | −11.37 % | 1.01x |
| mobile CIF | −9.99 % | −5.68 % | 1.04x |
| akiyo CIF | −25.07 % | −23.87 % | 0.98x |
| ducks_take_off 720p | −9.95 % | −2.47 % | 0.88x |
| park_joy 720p | −8.08 % | −4.53 % | 1.04x |
| **mean** | **−13.38 %** | **−9.58 %** | **0.96x** |

| `--preset slow` vs x265 slow | BD PSNR-Y | BD PSNR-avg | CPU |
|---|---:|---:|---:|
| foreman CIF | −8.07 % | −7.08 % | 0.84x |
| mobile CIF | −6.76 % | −3.71 % | 0.87x |
| akiyo CIF | −19.27 % | −19.54 % | 0.66x |
| ducks_take_off 720p | −0.44 % | +2.67 % | 0.90x |
| park_joy 720p | −0.75 % | +1.68 % | 0.99x |
| **mean** | **−7.06 %** | **−5.19 %** | **0.93x** |

These tables are round 5's (2026-11-15, branch head on develop 603f7f1).
develop's binary measured beside them at 1.08x / 0.97x / 0.92x of x265
(veryfast / medium / slow), its CIF clips at veryfast 1.12–1.17x.

Round 4's tables (develop 211e0f0: veryfast −10.49 %, medium −11.33 %,
slow −2.31 %, with ducks +3.20 %, park_joy +3.70 % and mobile +0.91 % at
slow) were measured each preset back to back with develop's binary, which
measured 1.10x / 0.98x / 0.79x
(veryfast / medium / slow) in the same session against round 3's 1.27x /
1.05x / 1.09x and the user's 1.22x / 1.05x / 1.13x: the `--jobs 4` CPU
column moves by ±0.05x per clip with the machine's load (foreman's
veryfast 1.35x here is one such outlier). The steadier ratio is
interleaved one-thread runs of the two binaries on the same encodes, two
repetitions: round 4 against develop 0.90x at veryfast (CIF 0.94–1.00x,
720p 0.89–0.90x), 0.98x at medium, 1.15x at slow — which puts the means at
about 0.99x / 0.96x / 0.91x of x265 on this session's develop figures.

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

The fourth round (2026-11-15). x265 as a black box first: added one at a
time to x265 medium on ducks, park_joy and mobile, `--rdoq-level 2`
bought −7.9 % (ducks −12.8 %), `--rect --limit-modes` −4.0 % (mobile
−7.7 %), `--no-early-skip` −2.8 %, four references −0.4 %, the star search
and `--subme 3` nothing; taken away from x265 slow, one reference cost
+4.2 % (mobile +12.0 %), RDOQ level 1 +7.7 %, no rectangular partitions
+3.9 %. Rivet against x265 with matched structure located the slow gap in
inter pictures, not intra: intra-only rivet is −7.8 % (ducks), −2.4 %
(park_joy), −10.8 % (mobile) against x265 slow `--keyint 1`, but P-only at
a constant quantiser +7.5 / +3.6 / +1.4 %; with RDOQ off on both sides
ducks is +2.4 %, so most of what is left there is what RDOQ buys x265
and not rivet. x265's frame quantisers (its CSV, crf 28) also showed its
intra picture at the P pictures' quantiser on the 720p clips and 1.5–4
below on the CIF ones, where rivet's was a fixed 3 below everywhere.

What it bought, in commit order:
- **An intra picture's offset from the lookahead propagation** (every
  tier): `h265_cutree`'s inheritance over sixteen pictures, at strength 1,
  sets only the intra picture's quantiser, 4.5 × (mean + 1.45) clamped to
  [−6, 0], and no unit carries a delta. The means order the clips the
  way the sweeps want them (foreman −2.1, mobile −2.0, akiyo −3.8,
  park_joy −0.8, ducks −1.4; fixed offsets 0 / −1 / −3 / −5 put ducks at
  +5.8 / +6.2 / +8.0 / +10.8 % and akiyo at −6.1 / −8.3 / −13.8 / −16.4 %):
  slow +0.58 → −0.59 %, medium −10.55 → −11.45 %, veryfast −11.88 →
  −12.52 %, CPU unchanged.
- **Iterative bi-prediction refinement** (slow): the list whose own search
  did worse is searched again, whole samples then the sub-sample rings,
  against twice the source less the other list's prediction, then the
  other list, a move kept only where the pair's exact SATD plus vector
  bits falls; tried only where the unrefined pair comes within 1.1x of
  the better one-list shape. Alone +0.58 → +0.01 %.
- **A Lagrangian per picture kind** (`Tune::lambda`, a thread-local scale
  every decision's λ reads): slow at 0.7 for every kind. Alone 0.8 gave
  −0.36 %, 0.65 −0.50 %, 0.5 +0.30 % (the CIF clips lose below 0.8, the
  720p ones want it lower still); with the two above, −0.59 → −2.30 %
  (−2.09 % at 0.8), and the refinement's gate kept that at 0.95x the CPU.
- **Veryfast's CPU**: an early split (`split_gate`: a unit above 8x8 whose
  2Nx2N search costs more than 2.5 SATD Lagrangians a sample, and is not
  a merge, goes straight to its children), the 8x8 transform split only
  where the whole TU kept four luma levels or more (`tu_split_nz`, also
  at medium), and three of the five merge candidates scored: −12.52 →
  −10.49 % at 0.89x the CPU by interleaved one-thread runs (medium:
  −11.45 → −11.33 % at 0.98x).
- SAO band statistics tallied per row in plain integers, byte-identical.

Tried and dropped this round: symmetric partitions at slow (mobile
+2.82 → +1.06 %, park_joy +6.70 → +5.32 %, at 1.8x x265's CPU); RDOQ
priced at the live contexts (P-only ducks +7.5 → +9.4 %, mobile +1.4 →
+0.5 %); RDOQ's own Lagrangian at 0.8 / 0.6 of the decisions' (slow −0.15
/ −0.08 %, subsumed by the per-picture Lagrangian); slow's Lagrangian by
kind, non-reference B at 0.5 (−1.86 %) or P and B at 0.6 (−2.03 %), against
0.7 everywhere (−2.30 %); at veryfast the early split at 1.5 (−10.02 %,
ducks −3.87 %), on 32x32 and 64x64 units only (−12.02 % for 4 %), on 64x64
units only (never fires: their searches are merges), no unit coded whole at
64x64 (−8.93 % at no saving), SAO off in reference pictures too (−10.29 %
for 3.5 %), quarter-sample refinement on a cross (mobile −6.97 → −4.11 %
for 2 %), merge candidates no longer seeding a unit that has its parent's
vector (no saving); the early split at medium (ducks −5.6 → −2.6 %).

The fifth round (2026-11-15), in commit order, each figure the mean
BD-rate PSNR-Y against x265 at that preset:
- **RDOQ weighs every block against coding nothing.** The last position
  had been forced to a level, so a block whose one significant
  coefficient sat just above half a step was always coded — which is why
  RDOQ had lost on inter chroma. With that test (the coded block flag's
  context named by the caller), the coefficient group flags priced in the
  last-position search and an empty group's significance flags not, and
  inter chroma quantised by RDOQ: slow −2.31 → −4.75 % (intra units'
  test alone −3.42 → −4.75 %), medium −11.33 → −14.25 %, veryfast −10.49 →
  −13.76 %. Inter chroma by RDOQ is now a gain (slow −2.56 → −3.28 %).
- **More than one reference per list in B slices** (`Tune::b_refs`):
  RefPicList0 the past references and RefPicList1 the future ones,
  nearest first; the header's active counts and RPS follow from them,
  `ref_idx_l0/l1` are coded where a list has more than one entry, each
  list keeps the reference whose vector and `ref_idx` cost least, and
  merge, AMVP and TMVP read the decoder's own derivations over the full
  lists (one per list under a weighted table). Slow, two per list, the
  further one searched below the top of the tree only where the enclosing
  unit chose it: −4.75 → −5.51 % (mobile −1.08 → −5.02 %) at 1.06x the
  CPU; unlimited −5.75 % at 1.17x, four per list −5.71 % at 1.09x. Medium
  gained −0.4 % for 5–9 % and keeps one.
- **Sign hiding weighs bins too**: the level moved is the one whose
  distortion plus 0.6 λ times a rough change in bins (3 crossing zero,
  1 crossing one, 0.5 above) grows least. Slow −5.51 → −5.85 % (ducks
  +0.43 → −0.03 %).
- **Writers generic over `BinEnc`** (`hevc::ctu`, `hevc::residual`, the
  CU writers), every rate a decision prices running through the
  fixed-point `CabacEstimator`: byte-identical, slow 3.4 % less CPU.
- **RDOQ's last position from 0.7 of a step** (`Tune::rdoq_last`; 0.85 at
  veryfast): the positions between were nearly always zeroed, and weighing
  them was most of the quantiser's time. Slow −5.85 → −6.00 % at 0.98x
  the CPU, medium −14.40 → −14.45 % at 0.94x, veryfast −13.91 → −14.09 %
  at 0.98x (0.85: veryfast −13.91 % at 0.96x; slow at 0.85 −5.10 %).
- **Slow's partitions, searched from the 2Nx2N unit's motion**
  (`Tune::part_seeded`): each unit of a 2NxN or Nx2N shape searches only
  the reference per list the 2Nx2N unit chose, from the vector it found
  there and the predictors, with no widening rings and no bi refinement;
  shapes are tried only where the 2Nx2N unit is not a merge, and the
  loser coded too only within 0.1 of the winner; a skip cheaper than 0.5
  λ per sample ends the recursion. −6.00 → −7.06 % (foreman −5.9 →
  −8.1 %, park_joy +0.4 → −0.8 %) at 0.93x x265's CPU. On the way:
  partitions searched from scratch −8.23 % at 2.09x x265's CPU; seeded
  −7.73 % at 1.65x; behind the merge gate −6.99 % at 1.16x; partitions
  only from 16x16 −6.11 % (8x8 CUs' partitions carry most of the gain).
- **Veryfast's CPU**: the RDOQ threshold above, two SAO edge classes of
  four (BD unchanged, 1.3 %), the early split at 2.0 rather than 2.5, and
  no allocation per prediction unit at any tier (the reference-list
  context's buffers reused, the per-reference and seen-candidate lists
  arrays, the flat-block proxy reading one row at stride zero; 1.3 %,
  byte-identical): 1.08x → 0.96x of x265 at −13.38 %.

Tried and dropped this round (slow unless named): RDOQ at the live
contexts again (−5.83 % against −5.51 %: the CIF clips 0.3–0.7 better,
ducks and park_joy 0.1–0.3 worse); RDOQ's chroma Lagrangian scaled by the
chroma weight (PSNR-Y 0.2 worse, PSNR-avg 0.1 better); a zero candidate
for levels rounding to 3 (identical); RDOQ's Lagrangian at 0.85 / 1.2
(−4.70 / −4.57 % against −4.75 %); the slow Lagrangian at 0.6 / 0.85 /
0.8 (−5.17 / −5.89 / −7.19 % against −5.51 / −5.51 / −7.06 %: the CIF clips
gain, ducks loses); the inter transform split up to 32x32 (−5.72 % at
1.07x); the 8x8 split only above four levels at slow (720p 0.3 worse);
the SAO Lagrangian at 0.6 / 1.5 of its own (−6.34 / −7.02 % against
−6.99 %); widening rings only at the top depths (park_joy −0.8 → −0.5 %);
the partition gate by cost rather than by merge (no better); four B
references per list on top of the partitions (−7.25 % at 1.04x). At
veryfast: no RDOQ in inter units (−9.48 %, ducks +0.59 %, at 0.94x of
develop's CPU); no inter transform split (−12.04 %, mobile −5.4 %, for
6 %), only in reference pictures (no saving); a higher split level count
(no saving); the half-sample phases alone for CIF references (1.14x:
every quarter phase interpolated on demand costs more than the planes);
two merge candidates (no saving); an all-zero test at 0.7 of a step
before RDOQ (0.5 %).

What is left, and why. Every clip now beats x265 at every preset:
veryfast −13.4 %, medium −14.5 %, slow −7.1 %, at 0.96x / 0.92x / 0.93x
of its CPU in the mean. The thin margins are slow's 720p clips (ducks
−0.44 %, park_joy −0.75 %): with RDOQ's block-empty test, two B
references and the partitions in, what x265 slow still has that rivet
does not is a transform split above 8x8 CUs (x265's inter TU depth 1;
rivet's costs 7 % for −0.2 %), four B pictures with an adaptive GOP, and
its partition search at full strength (the merge gate gives back 0.8 % on
the 720p clips for 0.25x of x265's CPU). The CPU per clip still runs
above x265's on veryfast's CIF clips and park_joy (1.01–1.04x): at
veryfast eight-tenths of the time is the quadtree's decisions, a third of
that the residual coding (RDOQ, the transforms, the counted rates) at
every node, which x265 veryfast avoids with `rd=2` and no RDOQ.

Earlier rounds' summary of the same, kept for its numbers: veryfast's
remaining CPU was the full RD decision at every quadtree node (residual,
RDOQ and pricing of each candidate, ~40 % of its time), SAO decisions in
the reference pictures (~8 %) and intra in inter pictures (~9 %); dropping
the 8x8 transform split there buys 6 % for 1.4 % BD (mobile −7.1 → −3.3 %),
quarter-sample refinement on a cross 1.3 % for 1.1 %, SAO in reference
pictures off 11 % for 2.4 % — none kept. Slow's spare CPU in round 3
found nothing to buy: the split transform tried in 16x16 and 32x32 CUs
(+0.57 / +0.48 % at 1.06 / 1.11x), intra transform splits in inter
pictures (+0.57 %), a second widening ring (+0.57 %), the losing
merge/AMVP shape coded too and the cheaper kept (+0.56 % at 1.2x),
quantiser offsets I −4 / B +1, +3, +5 (+0.48 / +0.84 / +0.92 / +0.93 %).

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
(+0.62 % at 2.0, −2.02 % at 1.0). Round 4 turns on its frame-level part
for the intra picture alone (above), which is the one place the fixed
offsets were wrong by content.

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
"H.264 compression against x264" above: 0.76x to 1.35x x264's
single-thread time at the same-named preset, 1.04x to 1.15x the time x264
needs for the same BD-rate (round 10B), the gap in rivet's trellis,
candidate pricing, residual coding and sub-sample SATD rather than in how
much it searches.

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
