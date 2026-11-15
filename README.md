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
| **H.264** | Baseline / Main / High / High 10 / High 4:2:2 / High 4:4:4 Predictive / CAVLC 4:4:4 Intra (and the Intra profiles): frames, field pictures (PAFF — field / frame reference lists and marking, field POCs, colocated field / frame mapping) and MBAFF (macroblock-adaptive frame / field: pair-wise neighbour derivation, field-scan and field-context entropy coding, mixed frame / field prediction and direct-mode mapping, mixed-edge deblocking), 4:0:0 / 4:2:0 / 4:2:2 / 4:4:4 at 8–14-bit, separate colour planes, lossless (transform bypass), CAVLC + CABAC, I/P/B, spatial + temporal direct, explicit + implicit weighting, 8x8 transform, scaling matrices, PCM, MMCO, multi-slice in any order (ASO), slice groups (FMO, all seven map types), SP / SI slices (Extended profile), frame-num gaps, all three POC types, deblocking, VUI reorder hints, the old-x264 4:4:4 CABAC quirk | unequal luma / chroma bit depths, data partitioning, SP / SI outside the Extended profile's shape (CABAC, 4:2:2 / 4:4:4, > 8-bit, 8x8 transform) |
| **H.265** | Main / Main 10 / Main 12 and the format range extensions — Main 4:2:2 / 4:4:4 (10, 12, 16 Intra), the High Throughput 4:4:4 (16 Intra) and Monochrome profiles (4:0:0 / 4:2:0 / 4:2:2 / 4:4:4, 8–16-bit, unequal luma / chroma depths): CTB 16–64, AMP, transform skip (any size, rotation, single-context), scaling lists, sign hiding, PCM, `cu_transquant_bypass` (lossless), cu_qp_delta, chroma QP offset lists, cross-component prediction, implicit / explicit RDPCM, persistent Rice adaptation, high-precision weighted-prediction offsets, intra smoothing disabling, extended precision processing, CABAC bypass alignment, tiles, WPP (and both together), dependent slice segments, merge / AMVP / TMVP, explicit weighting, deblocking, SAO, long-term references, CRA / BLA / RASL handling, `pic_output_flag`, `no_output_of_prior_pics`, decoded-picture-hash SEI verification (`H26X_VERIFY_HASH=1`) | separate colour planes, SCC, multi-layer |

Both decoders are **bit-exact**. H.264 passes **all 204** JVT conformance
bitstreams (AVCv1 + FRExt — every profile set, every field-picture and MBAFF
stream, the three FMO / ASO streams and the two SP streams) against the
suite's reconstructed YUV (the JM reference decoder's output where the zip
ships none), and **all 38** JVT professional-profile bitstreams (High 10 /
4:2:2 / 4:4:4 Intra, CAVLC 4:4:4 Intra, High 4:4:4 Predictive at up to
14-bit, thirteen of them coded as separate colour planes), which ship no
reference output and are checked against the JM reference decoder. Decoding
is deterministic across thread counts (every suite stream decoded on 1 and
12 threads gives the same bytes). It matches JM on the workspace fixtures too
(53 x264 / x265 encodes: CAVLC/CABAC, B-pyramids, weighting, 8x8, slices,
CQM, 10-bit, 4:2:2, 4:0:0, 4:4:4, lossless, interlaced). H.265 passes **147
of the 147** JCT-VC HEVC_v1 conformance bitstreams and **all 49** RExt
bitstreams against the suite's own MD5s (HM's output where a zip ships
none) — among them 16-bit, extended precision, CABAC bypass alignment and
unequal luma / chroma bit depths — and matches HM on the x265 feature
fixtures; every HM stream's decoded-picture-hash SEI is checked as well.
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
the B pyramid and the lookahead):

- **Motion**: the three half-sample planes of each reference are built once
  (`h264_hpel.rs`), so every quarter-sample candidate is a plane read or one
  rounded average — bit-exact with the decoder's kernels at all sixteen
  positions (tested). Candidates are the median, skip and zero vectors, the
  coded neighbours' and the lookahead's half-resolution vectors; a hexagon
  (veryfast: small diamond; slow: uneven multi-hexagon) descent on SAD plus
  lambda times the vector's rate, then half- and quarter-sample refinement
  on SATD plus rate. P pictures search up to 3 (slow 4) references.
- **Decision**: shapes, B directions and intra modes are estimated at SATD
  plus rate; skip, the best one to three inter shapes at both transform
  sizes and the best intra type are then coded and compared on
  `SSD + lambda * R`, with `R` priced by the picture's own CABAC writer from
  its live context states (`CabacEncoder::estimating`) — no table of what
  syntax "usually" costs. Early skip when the skip prediction quantises to
  nothing and its error is small.
- **Quantisation**: the chosen macroblock is re-quantised by a trellis over
  each block's level choices, priced from the live contexts
  (`h264_trellis.rs`).
- **Structure**: a B pyramid (the middle B a reference; a P picture's list 0
  reordered to its anchor), and a lookahead (`h264_lookahead.rs`) that
  propagates half-resolution intra/inter cost estimates back through each
  picture's vectors and codes reused reference macroblocks finer
  (`--tree N`).

Measured by a parity script outside the repository: 30 frames of five clips, rivet at
QP 22/27/32/37 against x264 0.165 `--preset P --tune psnr` at CRF 18/23/28/33,
one thread, both decoded by ffmpeg and compared raw; BD-rate on PSNR-Y and
PSNR-avg (negative: rivet needs fewer bits). Time is the ratio of CPU cycles
of interleaved runs on one loaded Apple M-series machine (the CPU-second
ratios of the parity run, taken four encodes at a time, agree within 0.1x):

| `--preset` | clip | BD-rate Y | BD-rate avg | time vs x264 |
|---|---|---:|---:|---:|
| veryfast | foreman CIF | +6.49 % | +6.28 % | 3.28x |
| | mobile CIF | -4.43 % | -6.00 % | 4.09x |
| | akiyo CIF | -1.09 % | +0.47 % | 3.55x |
| | ducks_take_off 720p | -9.97 % | -12.67 % | 3.85x |
| | park_joy 720p | -3.18 % | -3.49 % | 4.12x |
| | **mean** | **-2.44 %** | **-3.08 %** | **3.95x** |
| medium | foreman CIF | +7.73 % | +7.70 % | 2.12x |
| | mobile CIF | +8.72 % | +7.41 % | 2.39x |
| | akiyo CIF | +0.62 % | +1.71 % | 2.34x |
| | ducks_take_off 720p | +4.69 % | -0.11 % | 2.29x |
| | park_joy 720p | +5.08 % | +4.55 % | 2.40x |
| | **mean** | **+5.37 %** | **+4.25 %** | **2.33x** |
| slow | foreman CIF | +10.83 % | +10.63 % | 2.09x |
| | mobile CIF | +11.01 % | +9.48 % | 1.88x |
| | akiyo CIF | +4.01 % | +5.19 % | 2.83x |
| | ducks_take_off 720p | +7.25 % | +1.29 % | 1.64x |
| | park_joy 720p | +5.79 % | +5.02 % | 1.70x |
| | **mean** | **+7.78 %** | **+6.32 %** | **1.70x** |

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

Tried and dropped, with what they measured (same corpus, BD-rate Y against
medium unless named):

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

Where the medium gap is, decomposed against x264 configured down to rivet's
tool set (measured just before rivet's P pictures took several references):
against constant-QP x264 (no macroblock tree) rivet without its lookahead
was +4.7 %, of which x264's references and weighted prediction were ~2.2 %
(`--ref 1 --weightp 0` left +2.5 %) — rivet's P references then recovered
1.0 of it, leaving B-picture references and per-partition reference choice;
x264's macroblock tree is worth ~3 % to x264 (CRF against `--qp`) and
rivet's propagation ~2 % to rivet; the remaining ~2.5 % is
per-macroblock decisions. Slow lacks what
x264's slow adds (sub-8x8 partitions in the analysis, RD refinement of
sub-sample vectors and intra modes, trellis during decisions, more B
references), so it trails further than medium. The time gap is
implementation efficiency, not search breadth: x264 does more searching
per macroblock in its assembly than rivet does in Rust with SIMD for the
distortion and transform kernels only — the CABAC pricing of candidates
(a quarter of the time), the candidates' residual coding (a fifth), the
search's own control flow and the trellis are scalar.

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
NxN at 8x8) where that is cheaper; rates from an estimating CABAC encoder;
sign data hiding. Slow adds rate-distortion optimised quantisation. Medium
and veryfast prune: a skip cheaper than 0.5 λ per sample (veryfast: every
skip) is not split, intra is coded only where its best mode's SATD cost
beats the inter search, angles are searched coarse to fine, and pictures
nothing references take the cheapest settings.

Measured with `parity.py` (30 frames, QP 22/27/32/37 against x265 `--tune
psnr --crf 18/23/28/33`, x265 one thread; BD-rate, negative = fewer bits
than x265; CPU = summed child CPU seconds, rivet / x265, at `--jobs 3`,
indicative — a `--jobs 1` medium run read 1.62x where `--jobs 4` read 1.64x):

| `--preset medium` vs x265 medium | BD PSNR-Y | BD PSNR-avg | CPU |
|---|---:|---:|---:|
| foreman CIF | −10.84 % | −10.21 % | 1.32x |
| mobile CIF | −1.44 % | −0.50 % | 1.08x |
| akiyo CIF | −21.09 % | −19.54 % | 1.37x |
| ducks_take_off 720p | +11.15 % | +9.67 % | 0.97x |
| park_joy 720p | +5.49 % | +4.90 % | 1.26x |
| **mean** | **−3.35 %** | **−3.14 %** | **1.11x** |

| `--preset veryfast` vs x265 veryfast | BD PSNR-Y | BD PSNR-avg | CPU |
|---|---:|---:|---:|
| foreman CIF | −9.38 % | −8.51 % | 1.48x |
| mobile CIF | −0.19 % | +1.12 % | 1.29x |
| akiyo CIF | −21.12 % | −19.65 % | 1.43x |
| ducks_take_off 720p | +3.44 % | +2.81 % | 1.12x |
| park_joy 720p | +1.79 % | +1.25 % | 1.43x |
| **mean** | **−5.09 %** | **−4.60 %** | **1.27x** |

| `--preset slow` vs x265 slow | BD PSNR-Y | BD PSNR-avg | CPU |
|---|---:|---:|---:|
| foreman CIF | +4.52 % | +2.70 % | 1.32x |
| mobile CIF | +13.90 % | +13.63 % | 1.64x |
| akiyo CIF | −11.28 % | −11.35 % | 1.55x |
| ducks_take_off 720p | +10.59 % | +12.02 % | 1.23x |
| park_joy 720p | +12.07 % | +12.83 % | 1.48x |
| **mean** | **+5.96 %** | **+5.97 %** | **1.36x** |

Where it started (the same harness, `--bframes 3 --refs 4 --sao` at constant
QP): +46.7 % against medium at 2.31x the CPU, +65.7 % against slow, +40.5 %
against veryfast. The steps, mean BD-rate against medium: the SATD
Lagrangian +51.3 → +29.8 (CIF clips); per-kind quantiser offsets and the
pyramid +1.85; dropping residual by RD +0.35; the priced search −0.49;
intra by RD −3.46 (and NxN −4.40); sign hiding −6.06 at 1.62x; the medium
pruning −3.35 at 1.1x.

What is left, and why. The CIF means carry the tiers: akiyo and foreman
gain from the I-picture offset and intra decisions, while the 720p clips
trail. x265's own ablation (black box) says where: without its lookahead
quantiser propagation (`--no-cutree`) x265 medium loses 21.9 % on ducks,
14.3 % on park_joy and 4.1 % on foreman; `--ctu 32` costs it nothing
(−1.2 to +0.7 %), TMVP 0.9–2.7 %, weighted prediction 0. rivet has its own
propagation (`h265_cutree`, `--tree N`), and as built it loses (medium,
strength 2.0: −3.31 → +2.58 %; 1.0: −1.57 %; with the I offset dropped,
−2.87 / +0.04 %; 0.5: −3.04 %), so no tier turns it on: its model or its
application is wrong somewhere not yet found, and that is the largest
single gap. Slow's further gap is tools x265 slow has and rivet lacks or
has slowly: rectangular / asymmetric partitions (`--parts sym` gained
nothing at 1.9x the CPU), TMVP, RD refinement of modes. The CPU gap is
implementation: candidate residuals and their CABAC pricing (a fifth to a
quarter of the time), SAO decisions (9–13 %), the intra trials.

Tried and dropped (medium unless named): an early skip (best merge coded
first, kept when nothing survives) −0.5 % BD for 0.99x; diamond sub-sample
patterns −1.2 to −2.0 % for 0.95–0.98x; search widening rings −0.2 % for
1.06x (kept at slow); limiting references to the parent's −0.9 % for 0.96x;
two references instead of three −0.07 % for 0.94x (kept); a flat GOP at its
best offsets +12.0 %, pyramids of 5 and 7 +6.3 / +4.7 % against +1.85 % at 3;
RDOQ at medium −2.2 % BD for 1.31x CPU (slow only).

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
block whose last coefficient is three or more. The record, with the
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
"H.264 compression against x264" above: 1.7x to 3.9x x264's single-thread
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
