#!/bin/bash
# make_streams.sh — generate the `hevc_scp` suite: HEVC streams coded as
# separate colour planes (separate_colour_plane_flag = 1).
#
# No HEVC profile allows the flag (A.3.5 - A.3.8 all require it to be 0),
# so no encoder writes it, no conformance zip carries it and HM-18.0 does
# not decode it. Its decoding process is fully specified all the same
# (8.1.3: each colour plane is decoded "as if only a CVS with monochrome
# colour format with that particular value of colour_plane_id would be
# present"), and that gives each stream here a right answer that does not
# depend on this decoder: x265 encodes the Y, Cb and Cr planes of a 4:4:4
# synthetic clip as three monochrome streams with one picture structure,
# tools/make_separate_planes.py splices them into one separate-plane
# stream, and the reference output is HM's decodes of the three monochrome
# streams, interleaved picture by picture (Y, Cb, Cr) — written as
# <name>/<name>.md5, the MD5 of that YUV. Every picture also carries a
# three-component decoded picture hash SEI (run_conf.sh decodes with
# H26X_VERIFY_HASH=1 and fails a stream that mismatches one).
#
#   X265=path         x265 with 8-, 10- and 12-bit support (default: on PATH)
#   TAPPDECODER=path  HM's TAppDecoder (default: on PATH)
#   TOOLS=dir         the repository's tools/ (default ../..)
#
# Usage: make_streams.sh   (writes streams/<name>/ here)
set -u
cd "$(dirname "$0")"
HERE=$(pwd)
TOOLS=${TOOLS:-$(cd ../.. && pwd)}
X265=${X265:-$(command -v x265)}
HM=${TAPPDECODER:-$(command -v TAppDecoder)}
[ -x "$X265" ] || { echo "make_streams.sh: no x265 (set X265)" >&2; exit 2; }
[ -x "$HM" ] || { echo "make_streams.sh: no TAppDecoder (set TAPPDECODER)" >&2; exit 2; }
WORK=$HERE/gen
mkdir -p "$WORK" streams
fail=0
W=200
H=116   # not a multiple of 8: the conformance window is signalled

# make NAME DEPTH FRAMES [--interleave] -- X265 OPTIONS...
make() {
  local name=$1 depth=$2 frames=$3 il=""
  shift 3
  if [ "$1" = --interleave ]; then il=--interleave; shift; fi
  [ "$1" = -- ] && shift
  local fmt=444
  [ "$depth" != 8 ] && fmt=444p$depth
  local src="$WORK/src_${depth}_${frames}.yuv"
  [ -s "$src" ] || python3 -I "$TOOLS/synth_source.py" motion --size ${W}x$H --frames "$frames" --format $fmt \
    --noise $((depth - 8)) "$src" >&2
  local bps=1
  [ "$depth" -gt 8 ] && bps=2
  # The three planes as monochrome sources.
  python3 -I - "$src" "$WORK/$name" $((W * H * bps)) <<'PY'
import sys
src, stem, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = open(src, "rb").read()
for c, p in enumerate(("y", "cb", "cr")):
    open(f"{stem}_{p}.yuv", "wb").write(b"".join(d[f * 3 * n + c * n:f * 3 * n + (c + 1) * n] for f in range(len(d) // (3 * n))))
PY
  local p
  for p in y cb cr; do
    "$X265" --input "$WORK/${name}_$p.yuv" --input-res ${W}x$H --fps 25 --input-csp i400 --input-depth "$depth" \
      --output-depth "$depth" --qp 30 --keyint 8 --min-keyint 8 --no-scenecut --bframes 3 --b-adapt 0 \
      --no-open-gop --no-weightp --no-weightb --frame-threads 1 --hash 1 --log-level error --no-progress "$@" \
      -o "$WORK/${name}_$p.265" || { echo "FAIL $name: x265"; fail=1; return; }
    "$HM" -b "$WORK/${name}_$p.265" -o "$WORK/${name}_$p.ref.yuv" -d 0 > /dev/null \
      || { echo "FAIL $name: HM"; fail=1; return; }
  done
  mkdir -p "streams/$name"
  python3 -I "$TOOLS/make_separate_planes.py" "$WORK/${name}_y.265" "$WORK/${name}_cb.265" "$WORK/${name}_cr.265" \
    "streams/$name/$name.bit" $il || { echo "FAIL $name: splice"; fail=1; return; }
  # The expected output: HM's three monochrome decodes, plane by plane per picture.
  python3 -I - "$WORK/$name" $((W * H * bps)) "streams/$name/$name.md5" <<'PY'
import hashlib, sys
stem, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
planes = [open(f"{stem}_{p}.ref.yuv", "rb").read() for p in ("y", "cb", "cr")]
frames = len(planes[0]) // n
assert all(len(p) == frames * n for p in planes), "the planes decoded to different picture counts"
m = hashlib.md5()
for f in range(frames):
    for p in planes:
        m.update(p[f * n:(f + 1) * n])
open(out, "w").write(f"{m.hexdigest()}  {stem.rsplit('/', 1)[-1]}.yuv ({frames} frames, HM monochrome decodes)\n")
PY
  printf "%-28s %7d bytes  %s\n" "$name" "$(wc -c < "streams/$name/$name.bit" | tr -d ' ')" "$(cut -c1-32 "streams/$name/$name.md5")"
}

# x265 needs a thread pool for WPP; `--pools 1` keeps the output independent
# of the machine's core count.
make scp_8bit 8 20 -- --pools 1 --no-wpp
make scp_8bit_wpp_slices_interleaved 8 20 --interleave -- --pools 2 --wpp --slices 2 --ctu 16 --max-tu-size 16
make scp_10bit_wpp_slices 10 12 -- --pools 2 --wpp --slices 3 --ctu 32
make scp_12bit 12 12 -- --pools 1 --no-wpp --ctu 16 --max-tu-size 16
rm -rf "$WORK"
[ "$fail" = 0 ] && echo "made $(ls -d streams/*/ | wc -l | tr -d ' ') streams" || { echo "SOME STREAMS FAILED"; exit 1; }
