#!/bin/bash
# Every conformance suite (JVT AVCv1+FRExt, JVT professional profiles, JCT-VC
# HEVC_v1, RExt, SCC, MV-HEVC and SHVC, and the generated h264_jm and
# hevc_scp) at once, from a frozen copy of the current h26xdec
# build (so a rebuild does not disturb the run). Prints each suite's tally.
cd "$(dirname "$0")"
cp ../../release/examples/h26xdec.exe ../../release/examples/h26xdec_conf.exe || exit 1
J=${JOBS:-6}
SUITES="h264 h264_pp hevc hevc_rext hevc_scc hevc_mv hevc_shvc h264_jm hevc_scp"
for s in $SUITES; do
  JOBS=$J bash $s/run_conf.sh > $s/run_all.log 2>&1 &
done
wait
for s in $SUITES; do
  printf "%-10s %s\n" "$s" "$(tail -1 $s/run_all.log)"
  grep -E "^FAIL|^NOSTREAM" $s/run_all.log
done
