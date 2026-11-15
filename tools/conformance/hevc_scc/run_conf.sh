#!/bin/bash
# Run every JCT-VC SCC (screen content coding extensions) conformance stream
# through h26xdec and classify it. The suite's own whole-YUV MD5 is the
# reference when the zip carries one; otherwise the per-frame MD5s of HM's SCC
# build (HM-16.21+SCM-8.8, run as a program: TAPPDECODER_SCC, else
# TAPPDECODER), generated on first use.
#   PASS         bit-exact
#   UNSUPPORTED  the decoder refused it (Error::Unsupported)
#   FAIL         mismatch or bitstream error
# Streams run JOBS at a time (default 8), THREADS per decoder (default 4).
# Usage: run_conf.sh [name-filter]
cd "$(dirname "$0")"
DEC=${DEC:-$(cd ../../../release/examples && pwd)/h26xdec_conf.exe}
JOBS=${JOBS:-8}
# Two callers running at once must not share a scratch directory or a
# decoder copy: a run that quietly tested somebody else's binary still
# reports green.
OUT=${OUT:-out}
# The reference decoder (JM / HM, via ref_decode.py), for streams whose
# zip carries no reference output. REF_DECODE names the script: run from
# the repository it is ../../ref_decode.py; from a copy under
# $H26X_WORK/conf, point it at the repository's tools/ref_decode.py.
REF_DECODE=${REF_DECODE:-$(cd ../.. && pwd)/ref_decode.py}
TAPPDECODER=${TAPPDECODER_SCC:-$TAPPDECODER}
export DEC OUT REF_DECODE TAPPDECODER H26X_VERIFY_HASH=1 H26X_THREADS=${THREADS:-4}
mkdir -p refs "$OUT"
one() {
  d=$1
  name=$(basename "$d")
  MD5REC="$OUT/md5parts-$$.tmp"
  bs=$(find "$d" -maxdepth 4 -type f \( -name "*.bit" -o -name "*.bin" -o -name "*.265" -o -name "*.hevc" \) 2>/dev/null | head -1)
  [ -z "$bs" ] && { echo "NOSTREAM    $name"; return; }
  cp "$bs" "$OUT/$name.265" 2>/dev/null
  "$DEC" "$OUT/$name.265" "$OUT/$name.yuv" > "$OUT/$name.mine" 2> "$OUT/$name.err"
  status=$?
  n_mine=$(wc -l < "$OUT/$name.mine" | tr -d " ")
  # Suite MD5 (of the whole decoded YUV), if any: every 32-hex string in the
  # suite's md5 files (bitstream md5s among them are harmless).
  suite_md5s=$(cat $(find "$d" -maxdepth 4 -type f \( -name "*.md5" -o -name "*md5*.txt" \) 2>/dev/null) 2>/dev/null | grep -io '[0-9a-f]\{32\}' | tr 'A-F' 'a-f' | sort -u)
  mine_md5=$(md5sum "$OUT/$name.yuv" | cut -c1-32)
  # An RGB stream (GBR coded: Y = G, Cb = B, Cr = R) whose suite md5 names
  # an .rgb file: that file holds each frame's planes as R, G, B (checked
  # on Slice_ACT_QP_Offsets_A, whose decode HM-SCM writes in the coded
  # order G, B, R like h26xdec). Compare in that order as well.
  if grep -qi '\.rgb' $(find "$d" -maxdepth 4 -type f -name "*.md5") 2>/dev/null && [ -s "$OUT/$name.mine" ]; then
    geo=$(head -1 "$OUT/$name.mine" | cut -d, -f4)
    rgb_md5=$(python3 -I -c '
import hashlib, sys
path, w, h, n = sys.argv[1], *map(int, sys.argv[2:])
d = open(path, "rb").read()
f = len(d) // n
p = f // 3
out = hashlib.md5()
for i in range(n):
    g, b, r = (d[i * f + k * p : i * f + (k + 1) * p] for k in range(3))
    out.update(r + g + b)
print(out.hexdigest())
' "$OUT/$name.yuv" ${geo%x*} ${geo#*x} "$n_mine")
    echo "$suite_md5s" | grep -q "^$rgb_md5$" && mine_md5=$rgb_md5
  fi
  if grep -q "unsupported" "$OUT/$name.err"; then
    echo "UNSUPPORTED $name: $(tail -1 $OUT/$name.err)"
  elif [ $status -eq 0 ] && echo "$suite_md5s" | grep -q "^$mine_md5$"; then
    echo "PASS        $name ($n_mine frames, suite md5)"
  else
    ref="refs/$name.ref.md5"
    [ -s "$ref" ] || python "$REF_DECODE" framemd5 "$bs" "$n_mine" > "$ref" 2> "$OUT/$name.referr" || rm -f "$ref"
    n_ref=$(cat "$ref" 2>/dev/null | wc -l | tr -d " ")
    match=$(paste <(awk '{print $2}' "$ref" 2>/dev/null) <(awk -F, '{print $5}' "$OUT/$name.mine") | awk '{ if ($1==$2 && $1!="") ok++ } END{print ok+0}')
    if [ $status -eq 0 ] && [ "$n_ref" = "$n_mine" ] && [ "$match" = "$n_ref" ]; then
      echo "PASS        $name ($n_ref frames, HM)"
    else
      echo "FAIL        $name ref=$n_ref mine=$n_mine match=$match suite=[$(echo $suite_md5s | tr '\n' ' ')] mine_md5=$mine_md5 $(tail -1 $OUT/$name.err | head -c 120)"
    fi
  fi
  [ $status -eq 0 ] && echo "$name $mine_md5" >> "$MD5REC"
  rm -f "$OUT/$name.265" "$OUT/$name.yuv"
}
export -f one
ls -d streams/*/ | { if [ -n "$1" ]; then grep -i -- "$1"; else cat; fi; } \
  | xargs -P "$JOBS" -I{} bash -c 'one "$1"' _ {} | sort > $OUT/results.txt
cat "$OUT"/md5parts-*.tmp 2>/dev/null | sort > "$OUT/md5s.txt"; rm -f "$OUT"/md5parts-*.tmp
cat $OUT/results.txt
echo "pass=$(grep -c '^PASS' $OUT/results.txt) fail=$(grep -c '^FAIL' $OUT/results.txt) unsupported=$(grep -c '^UNSUPPORTED' $OUT/results.txt)"
