#!/bin/bash
# P2-a L31 (audit C11): cross-binary neutrality of the memory restrictions and the allocation trace with the new switch OFF.
#
# Usage: l31_compare.sh <baseline ov_gpu_unit_tests> <new ov_gpu_unit_tests> [workdir]
# Both binaries run the DISABLED data-producer test l31_dump.DISABLED_corpus (passes/l31_dump_corpus_test.cpp; for the baseline the file is
# copied into a baseline tree, see the commented steps below) and the dump directories are diffed byte for byte:
#   <tag>.memdeps.txt  get_memory_dependencies_string() content (in_order and out_of_order configurations)
#   <tag>.trace.txt    allocation trace records (debug caps builds)
#   <tag>.out.txt      output values
#
# Building the baseline binary (heavy; pick one):
#   (a) git worktree:   git worktree add /tmp/ov_base 85dd03d2f0
#                       cp tests/unit/passes/l31_dump_corpus_test.cpp /tmp/ov_base/src/plugins/intel_gpu/tests/unit/passes/
#                       add "${CMAKE_CURRENT_LIST_DIR}/passes/l31_dump_corpus_test.cpp" to /tmp/ov_base/.../tests/unit/sources.cmake
#                       configure + build ov_gpu_unit_tests in a separate build dir with the options of build_RelWithDebInfo
#   (b) same tree:      git stash / checkout the baseline commit in a scratch clone of the build dir (not recommended: rebuild cascade twice)
set -u
BASE=${1:?baseline binary}; NEW=${2:?new binary}; W=${3:-/tmp/l31}
rm -rf "$W"; mkdir -p "$W/base" "$W/new"
OV_GPU_L31_DUMP_DIR="$W/base" "$BASE" --gtest_also_run_disabled_tests --gtest_filter='l31_dump.DISABLED_corpus' > "$W/base.log" 2>&1 || { echo "baseline run failed, see $W/base.log"; exit 2; }
OV_GPU_L31_DUMP_DIR="$W/new"  "$NEW"  --gtest_also_run_disabled_tests --gtest_filter='l31_dump.DISABLED_corpus' > "$W/new.log" 2>&1  || { echo "new run failed, see $W/new.log"; exit 2; }
n=$(ls "$W/base" | wc -l)
echo "files per side: $n (base) $(ls "$W/new" | wc -l) (new)"
if diff -r "$W/base" "$W/new" > "$W/diff.txt"; then
  echo "L31 PASS: all $n dump files are byte identical"
else
  echo "L31 FAIL: see $W/diff.txt"; head -40 "$W/diff.txt"; exit 1
fi
