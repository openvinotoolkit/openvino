#!/usr/bin/env bash
# Build + run a GPU probe host runner (the .cpp files that actually execute a kernel on the
# GPU and print the observed data layout, e.g. which (key,head) each lane/byte carries).
# Unlike compare_isa.sh (static ISA), these run on-device to VERIFY layouts / correctness.
#
# Usage: test/microbench/run_probe.sh verify_k_transform   # or probe_v_layouts
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
name="${1:?usage: run_probe.sh <verify_k_transform|probe_v_layouts>}"
src="$DIR/$name.cpp"; [ -f "$src" ] || { echo "no such probe: $src"; exit 1; }
bin="$DIR/dump/$name.bin"
mkdir -p "$DIR/dump"
g++ -DCL_TARGET_OPENCL_VERSION=300 "$src" -lOpenCL -o "$bin" 2>&1 | grep -iE 'error' && exit 1
"$bin"
