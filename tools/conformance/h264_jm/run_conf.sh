#!/bin/bash
# Run every generated `h264_jm` stream (make_streams.sh: slice data
# partitioning, unequal luma / chroma bit depths) through h26xdec and
# classify it against the JM reference decoder's per-frame MD5s — there is
# no suite output for these, JM is the reference for every one.
#   PASS         bit-exact
#   UNSUPPORTED  the decoder refused it (Error::Unsupported)
#   FAIL         mismatch or bitstream error
# Streams run JOBS at a time (default 8), THREADS per decoder (default 4).
# Usage: run_conf.sh [name-filter]
cd "$(dirname "$0")"
DEC=${DEC:-$(cd ../../../release/examples && pwd)/h26xdec_conf.exe}
JOBS=${JOBS:-8}
OUT=${OUT:-out}
REF_DECODE=${REF_DECODE:-$(cd ../.. && pwd)/ref_decode.py}
export DEC OUT REF_DECODE H26X_THREADS=${THREADS:-4}
mkdir -p refs "$OUT"
one() {
  d=$1
  name=$(basename "$d")
  MD5REC="$OUT/md5parts-$$.tmp"
  bs="$d/$name.264"
  [ -f "$bs" ] || { echo "NOSTREAM    $name"; return; }
  "$DEC" "$bs" "$OUT/$name.yuv" > "$OUT/$name.mine" 2> "$OUT/$name.err"
  status=$?
  n_mine=$(wc -l < "$OUT/$name.mine" | tr -d " ")
  mine_md5=$(md5sum "$OUT/$name.yuv" | cut -c1-32)
  if grep -q "unsupported" "$OUT/$name.err"; then
    echo "UNSUPPORTED $name: $(tail -1 $OUT/$name.err | head -c 100)"
  else
    ref="refs/$name.ref.md5"
    [ -s "$ref" ] || python3 "$REF_DECODE" framemd5 "$bs" "$n_mine" > "$ref" 2> "$OUT/$name.referr" || rm -f "$ref"
    n_ref=$(cat "$ref" 2>/dev/null | wc -l | tr -d " ")
    match=$(paste <(awk '{print $2}' "$ref" 2>/dev/null) <(awk -F, '{print $5}' "$OUT/$name.mine") | awk '{ if ($1==$2 && $1!="") ok++ } END{print ok+0}')
    if [ $status -eq 0 ] && [ "$n_ref" -gt 0 ] && [ "$n_ref" = "$n_mine" ] && [ "$match" = "$n_ref" ]; then
      echo "PASS        $name ($n_ref frames, JM)"
    else
      echo "FAIL        $name ref=$n_ref mine=$n_mine match=$match $(tail -1 $OUT/$name.err | head -c 120)"
    fi
  fi
  [ $status -eq 0 ] && echo "$name $mine_md5" >> "$MD5REC"
  rm -f "$OUT/$name.yuv"
}
export -f one
ls -d streams/*/ | { if [ -n "$1" ]; then grep -i -- "$1"; else cat; fi; } \
  | xargs -P "$JOBS" -I{} bash -c 'one "$1"' _ {} | sort > $OUT/results.txt
cat "$OUT"/md5parts-*.tmp 2>/dev/null | sort > "$OUT/md5s.txt"; rm -f "$OUT"/md5parts-*.tmp
cat $OUT/results.txt
echo "pass=$(grep -c '^PASS' $OUT/results.txt) fail=$(grep -c '^FAIL' $OUT/results.txt) unsupported=$(grep -c '^UNSUPPORTED' $OUT/results.txt)"
