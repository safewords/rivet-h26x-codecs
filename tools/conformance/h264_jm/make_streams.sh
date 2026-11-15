#!/bin/bash
# make_streams.sh — generate the `h264_jm` suite: H.264 streams for the
# coding tools no JVT conformance zip carries, made by the JM reference
# encoder (`lencod`, run as a program — its source is not read) and judged
# by the JM reference decoder (`ldecod`) in run_conf.sh.
#
#   dp_*   slice data partitioning (Extended profile, PartitionMode=1,
#          CAVLC): nal_unit_type 2 / 3 / 4, partitions B and C absent where
#          a slice has no residual of that category.
#   ud_*   unequal luma / chroma bit depths (High 10 / 4:2:2 / 4:4:4
#          profiles): every chroma format, both entropy coders, the 8x8
#          transform, B pictures with weighted prediction (chroma offsets
#          included: WPMethod=2 on a source whose chroma drifts), fields.
#
# The AVCv1 suite has no partitioned stream and no suite stream has unequal
# depths, so these are the only coverage either has; the reference is the
# reference decoder, frame by frame.
#
# Not here, and why: MBAFF and SP. lencod's MBAFF output does not decode in
# its own ldecod (not even unpartitioned, 8-bit, Main), and its SP output
# decodes in ldecod to other chroma than lencod's own reconstruction (which
# this decoder matches, as it matches the suite's SP streams), so neither
# has a reference to be checked against.
#
#   LENCOD=path   JM's lencod (default: lencod on PATH)
#   TOOLS=dir     the repository's tools/ (default ../..)
#
# Usage: make_streams.sh   (writes streams/<name>/<name>.264 here)
set -u
cd "$(dirname "$0")"
HERE=$(pwd)
# The repository's tools/ (synth_source.py): ../.. when run from the
# repository, TOOLS=<repo>/tools from a copy under $H26X_WORK/conf.
TOOLS=${TOOLS:-$(cd ../.. && pwd)}
LENCOD=${LENCOD:-$(command -v lencod)}
[ -x "$LENCOD" ] || { echo "make_streams.sh: no lencod (set LENCOD)" >&2; exit 2; }
WORK=$HERE/gen
mkdir -p "$WORK" streams
: > "$WORK/empty.cfg"   # every parameter on the command line, none from a file
fail=0

# A source of FRAMES frames, 16-bit words per sample (what lencod reads once
# any depth exceeds 8), luma at BDY bits and chroma at BDC, from the
# deterministic synthetic `motion` clip; CSHIFT=1 makes the chroma drift
# upwards picture by picture, which is what makes lencod code chroma
# weighted-prediction offsets.
source16() { # fmt bdy bdc frames cshift -> file
  local f="$WORK/src_${1}_${2}_${3}_${4}_${5}.yuv"
  if [ ! -s "$f" ]; then
    python3 -I "$TOOLS/synth_source.py" motion --size 176x144 --frames "$4" --format "${1}p12" --noise 4 "$WORK/tmp12.yuv" >&2
    python3 -I - "$1" "$2" "$3" "$5" "$WORK/tmp12.yuv" "$f" <<'PY'
import struct, sys
fmt, bdy, bdc, cshift, src, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == "1", sys.argv[5], sys.argv[6]
w, h = 176, 144
n = w * h
c = {"420": n // 4, "422": n // 2, "444": n}[fmt]
d = open(src, "rb").read()
fs = 2 * (n + 2 * c)
o = bytearray()
for f in range(len(d) // fs):
    fr = struct.unpack("<%dH" % (n + 2 * c), d[f * fs:(f + 1) * fs])
    for i, v in enumerate(fr):
        bd = bdy if i < n else bdc
        v >>= 12 - bd
        if i >= n and cshift:
            v = min((1 << bd) - 1, v + f * (1 << (bd - 4)))
        o += struct.pack("<H", v)
open(out, "wb").write(o)
PY
  fi
  echo "$f"
}

# enc NAME SOURCE FRAMES -p ARG... : one lencod run into streams/NAME/NAME.264.
enc() {
  local name=$1 src=$2 frames=$3; shift 3
  mkdir -p "streams/$name"
  (cd "$WORK" && "$LENCOD" -d empty.cfg -p InputFile="$src" -p SourceWidth=176 -p SourceHeight=144 \
    -p FramesToBeEncoded="$frames" -p OutputFile="$HERE/streams/$name/$name.264" -p ReconFile=rec.yuv \
    -p IntraPeriod=4 -p QPISlice=28 -p QPPSlice=30 -p QPBSlice=32 "$@" > "$WORK/$name.log" 2>&1)
  # lencod exits non-zero over a missing leaky-bucket file even when the
  # stream is complete; the stream itself is the test.
  if [ -s "streams/$name/$name.264" ]; then
    printf "%-26s %7d bytes\n" "$name" "$(wc -c < "streams/$name/$name.264" | tr -d ' ')"
  else
    echo "FAIL $name: $(grep -iE 'error|not rec|not supported' "$WORK/$name.log" | head -1)"
    fail=1
  fi
}

# Data partitioning: 8-bit 4:2:0.
python3 -I "$TOOLS/synth_source.py" motion --size 176x144 --frames 10 --format 420 "$WORK/src420.yuv" >&2
S8=$WORK/src420.yuv
dp=(-p ProfileIDC=88 -p LevelIDC=30 -p PartitionMode=1 -p SymbolMode=0)
enc dp_ippp "$S8" 10 "${dp[@]}" -p IntraPeriod=8
enc dp_ibbp_temporal "$S8" 10 "${dp[@]}" -p NumberBFrames=2 -p NumberReferenceFrames=3
enc dp_ibp_spatial "$S8" 10 "${dp[@]}" -p NumberBFrames=1 -p DirectModeType=1 -p NumberReferenceFrames=2
enc dp_weighted "$S8" 10 "${dp[@]}" -p NumberBFrames=1 -p NumberReferenceFrames=2 -p WeightedPrediction=1 -p WeightedBiprediction=1
enc dp_cip_refresh "$S8" 10 "${dp[@]}" -p ConstrainedIntraPred=1 -p RandomIntraMBRefresh=10
enc dp_field "$S8" 10 "${dp[@]}" -p PicInterlace=1
enc dp_paff "$S8" 10 "${dp[@]}" -p PicInterlace=2
enc dp_slices "$S8" 10 "${dp[@]}" -p SliceMode=1 -p SliceArgument=20

# Unequal bit depths.
ud() { # name fmt bdy bdc profile cshift -p...
  local name=$1 fmt=$2 bdy=$3 bdc=$4 prof=$5 cs=$6; shift 6
  local yf=1
  [ "$fmt" = 422 ] && yf=2
  [ "$fmt" = 444 ] && yf=3
  enc "$name" "$(source16 "$fmt" "$bdy" "$bdc" 8 "$cs")" 8 -p ProfileIDC="$prof" -p LevelIDC=40 -p YUVFormat=$yf \
    -p SourceBitDepthLuma="$bdy" -p SourceBitDepthChroma="$bdc" -p OutputBitDepthLuma="$bdy" \
    -p OutputBitDepthChroma="$bdc" "$@"
}
ud ud_420_10_8 420 10 8 110 0
ud ud_420_8_10 420 8 10 110 0
ud ud_420_9_10_cabac_t8 420 9 10 110 0 -p SymbolMode=1 -p Transform8x8Mode=1
ud ud_420_10_8_ibp_wp 420 10 8 110 0 -p NumberBFrames=1 -p NumberReferenceFrames=2 -p WeightedPrediction=1 \
  -p WeightedBiprediction=1 -p SymbolMode=1
ud ud_420_8_10_wp_chroma 420 8 10 110 1 -p NumberBFrames=1 -p NumberReferenceFrames=2 -p WeightedPrediction=1 \
  -p WeightedBiprediction=1 -p SymbolMode=1 -p IntraPeriod=0 -p WPMethod=2
ud ud_420_10_8_field 420 10 8 110 0 -p PicInterlace=1 -p SymbolMode=1
ud ud_422_10_9_cabac_t8 422 10 9 122 0 -p SymbolMode=1 -p Transform8x8Mode=1
ud ud_422_8_10 422 8 10 122 0
ud ud_444_8_12_cabac_t8 444 8 12 244 0 -p SymbolMode=1 -p Transform8x8Mode=1
ud ud_444_10_8 444 10 8 244 0
rm -rf "$WORK"
[ "$fail" = 0 ] && echo "made $(ls -d streams/*/ | wc -l | tr -d ' ') streams" || { echo "SOME ENCODES FAILED"; exit 1; }
