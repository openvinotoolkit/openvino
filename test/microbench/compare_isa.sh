#!/usr/bin/env bash
# Compile one or more microbench .cl files with ocloc for bmg, dump the GEN ISA (.asm)
# via IGC ShaderDumpEnable, and print an instruction-mix summary for each. Used to compare
# reformulations WITHOUT touching the real kernel (observer-effect-free). See the
# gpu-kernel-isa-dump + sdpa-ocl-int8-perf memories.
#
# Usage:
#   test/microbench/compare_isa.sh                 # compare all *.cl in this dir
#   test/microbench/compare_isa.sh k_dequant_float.cl k_dequant_half.cl
#
# Output ISA lands in test/microbench/dump/<name>/ (persistent, not /tmp).
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
DEV=bmg
OUT="$DIR/dump"

files=("$@")
if [ ${#files[@]} -eq 0 ]; then
  mapfile -t files < <(cd "$DIR" && ls *.cl 2>/dev/null)
fi

for f in "${files[@]}"; do
  name="$(basename "$f" .cl)"
  src="$DIR/$f"; [ -f "$src" ] || src="$DIR/$name.cl"
  d="$OUT/$name"; rm -rf "$d"; mkdir -p "$d"
  IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir="$d" \
    ocloc compile -file "$src" -device $DEV -out_dir "$d" >"$d/compile.log" 2>&1
  A=$(find "$d" -name '*.asm' | head -1)
  echo "==== $name ===="
  if [ -z "$A" ]; then echo "  COMPILE FAILED (see $d/compile.log)"; grep -iE 'error' "$d/compile.log" | head -3; continue; fi
  echo "  total instr: $(grep -cE '^\s*(\([WfF][^)]*\)\s*)?[a-z][a-z0-9_.]*\s+\(' "$A")"
  echo "  mov=$(grep -cE '^\s*(\(.*\))?\s*mov ' "$A")  mul=$(grep -cE '^\s*(\(.*\))?\s*mul ' "$A")  add=$(grep -cE '^\s*(\(.*\))?\s*add ' "$A")  dpas=$(grep -cE 'dpas' "$A")"
  echo "  mov by dst type: $(grep -E '^\s*(\(.*\))?\s*mov ' "$A" | grep -oE ':(hf|f|b|w|uw|d|ud)\b' | sort | uniq -c | tr '\n' ' ')"
  echo "  <2>-strided movs: $(grep -cE 'mov .*<2>' "$A")   shuffle/bcst: $(grep -cE 'shuffle|bcst' "$A")   {Compacted}: $(grep -cE '\{.*Compacted' "$A")"
  echo "  asm: $A"
done
