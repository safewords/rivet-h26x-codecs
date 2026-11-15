#!/bin/bash
# Run every generated separate-colour-plane stream (make_streams.sh)
# through h26xdec and classify it: the whole decoded YUV must have the MD5
# make_streams.sh recorded from HM's decodes of the three monochrome
# streams the stream was spliced from, and every picture's decoded picture
# hash SEI must match (H26X_VERIFY_HASH=1: a mismatch is a warning, and any
# warning fails the stream here).
#   PASS         bit-exact, every hash SEI matched
#   UNSUPPORTED  the decoder refused it (Error::Unsupported)
#   FAIL         mismatch, hash mismatch or bitstream error
# Streams run JOBS at a time (default 8), THREADS per decoder (default 4).
# Usage: run_conf.sh [name-filter]
cd "$(dirname "$0")"
DEC=${DEC:-$(cd ../../../release/examples && pwd)/h26xdec_conf.exe}
JOBS=${JOBS:-8}
OUT=${OUT:-out}
export DEC OUT H26X_VERIFY_HASH=1 H26X_THREADS=${THREADS:-4}
mkdir -p "$OUT"
one() {
  d=$1
  name=$(basename "$d")
  MD5REC="$OUT/md5parts-$$.tmp"
  bs="$d/$name.bit"
  [ -f "$bs" ] || { echo "NOSTREAM    $name"; return; }
  cp "$bs" "$OUT/$name.265"
  "$DEC" "$OUT/$name.265" "$OUT/$name.yuv" > "$OUT/$name.mine" 2> "$OUT/$name.err"
  status=$?
  n_mine=$(wc -l < "$OUT/$name.mine" | tr -d " ")
  mine_md5=$(md5sum "$OUT/$name.yuv" | cut -c1-32)
  ref_md5=$(cut -c1-32 "$d/$name.md5" 2>/dev/null)
  warnings=$(tail -1 "$OUT/$name.err" | grep -o '[0-9]* warnings' | cut -d' ' -f1)
  if grep -q "unsupported" "$OUT/$name.err"; then
    echo "UNSUPPORTED $name: $(tail -1 $OUT/$name.err)"
  elif [ $status -eq 0 ] && [ -n "$ref_md5" ] && [ "$mine_md5" = "$ref_md5" ] && [ "${warnings:-1}" = 0 ]; then
    echo "PASS        $name ($n_mine frames, HM monochrome decodes, hash SEIs)"
  else
    echo "FAIL        $name mine=$n_mine mine_md5=$mine_md5 ref=$ref_md5 warnings=${warnings:-?} $(tail -1 $OUT/$name.err | head -c 120)"
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
