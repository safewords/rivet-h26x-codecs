#!/bin/bash
# Run every stream of a JCT-VC multi-layer conformance suite (MV-HEVC or
# SHVC: this script serves both) through h26xdec and classify it on its BASE
# LAYER — what a single-layer decoder outputs from a multi-layer stream
# (every NAL unit with nuh_layer_id > 0 discarded). The reference is the
# suite's own MD5 of the layer-0 YUV (`<name>-L0.md5`, or the `_0.yuv`
# line of an MV-HEVC md5 file) when the zip carries one, else the HM
# reference decoder's per-frame MD5s — HM-18.0 decodes layer 0 only. A
# stream with no base layer of its own (an external, e.g. H.264, base:
# HYBRID_A; or layers above 0 only: INBLD_A) passes when neither decoder
# outputs a picture.
#   PASS         bit-exact base layer
#   UNSUPPORTED  the decoder refused it (Error::Unsupported)
#   FAIL         mismatch or bitstream error
# Streams run JOBS at a time (default 8), THREADS per decoder (default 4).
# Usage: run_conf.sh [name-filter]
cd "$(dirname "$0")"
DEC=${DEC:-$(cd ../../../release/examples && pwd)/h26xdec_conf.exe}
JOBS=${JOBS:-8}
OUT=${OUT:-out}
REF_DECODE=${REF_DECODE:-$(cd ../.. && pwd)/ref_decode.py}
export DEC OUT REF_DECODE H26X_VERIFY_HASH=1 H26X_THREADS=${THREADS:-4}
mkdir -p refs "$OUT"
one() {
  d=$1
  name=$(basename "$d")
  MD5REC="$OUT/md5parts-$$.tmp"
  bs=$(find "$d" -maxdepth 4 -type f \( -name "*.bit" -o -name "*.bin" \) 2>/dev/null | head -1)
  [ -z "$bs" ] && { echo "NOSTREAM    $name"; return; }
  cp "$bs" "$OUT/$name.265" 2>/dev/null
  "$DEC" "$OUT/$name.265" "$OUT/$name.yuv" > "$OUT/$name.mine" 2> "$OUT/$name.err"
  status=$?
  n_mine=$(wc -l < "$OUT/$name.mine" | tr -d " ")
  base_md5=$( (cat $(find "$d" -maxdepth 4 -type f -name "*-L0.md5") 2>/dev/null; \
               grep -h "_0\.yuv" $(find "$d" -maxdepth 4 -type f -name "*.md5") 2>/dev/null) \
             | grep -io '[0-9a-f]\{32\}' | tr 'A-F' 'a-f' | sort -u)
  mine_md5=$(md5sum "$OUT/$name.yuv" | cut -c1-32)
  if grep -q "unsupported" "$OUT/$name.err"; then
    echo "UNSUPPORTED $name: $(tail -1 $OUT/$name.err)"
  elif [ $status -eq 0 ] && [ -n "$base_md5" ] && echo "$base_md5" | grep -q "^$mine_md5$"; then
    echo "PASS        $name ($n_mine frames, suite layer-0 md5)"
  else
    ref="refs/$name.ref.md5"
    if [ ! -f "$ref" ]; then
      if python "$REF_DECODE" decode --codec h265 "$bs" "$OUT/$name.ref.yuv" > /dev/null 2> "$OUT/$name.referr" \
         && [ -s "$OUT/$name.ref.yuv" ]; then
        python "$REF_DECODE" framemd5 --codec h265 "$bs" "$n_mine" > "$ref" 2>> "$OUT/$name.referr" || rm -f "$ref"
      else
        # HM wrote no picture: the stream has no base layer to output.
        : > "$ref"
      fi
      rm -f "$OUT/$name.ref.yuv"
    fi
    n_ref=$(cat "$ref" 2>/dev/null | wc -l | tr -d " ")
    match=$(paste <(awk '{print $2}' "$ref" 2>/dev/null) <(awk -F, '{print $5}' "$OUT/$name.mine") | awk '{ if ($1==$2 && $1!="") ok++ } END{print ok+0}')
    if [ $status -eq 0 ] && [ -f "$ref" ] && [ "$n_ref" = 0 ] && [ "$n_mine" = 0 ]; then
      echo "PASS        $name (no base layer: 0 frames, HM 0)"
    elif [ $status -eq 0 ] && [ "$n_ref" = "$n_mine" ] && [ "$match" = "$n_ref" ]; then
      echo "PASS        $name ($n_ref frames, HM)"
    else
      echo "FAIL        $name ref=$n_ref mine=$n_mine match=$match suite=[$(echo $base_md5 | tr '\n' ' ')] mine_md5=$mine_md5 $(tail -1 $OUT/$name.err | head -c 120)"
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
