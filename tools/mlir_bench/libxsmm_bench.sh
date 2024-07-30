#!/bin/bash

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Runs MLP benchmarks using libxsmm.

die_syntax() {
  echo "Syntax: $0 [-B] [-D]"
  echo ""
  echo "  -B: Use bf16 data type"
  echo "  -D: Set model shapes to dynamic"
  exit 1
}

# Cmd-line opts
while getopts "BD" arg; do
  case ${arg} in
    B)
      DATA_TYPE="bf16"
      ;;
    D)
      IS_DYNAMIC=true
      ;;
    ?)
      echo "Invalid option: ${OPTARG}"
      die_syntax
      ;;
  esac
done

BENCH_RUNNER=xsmm_dnn_mlp

# Initial validation.
if ! [ "$(command -v ${BENCH_RUNNER})" ]; then
  echo "Missing benchmark runner ${BENCH_RUNNER}"
  exit 1
fi
if [ ${IS_DYNAMIC} ]; then
  echo "Dynamic shapes are not supported by ${BENCH_RUNNER}"
  exit 1
fi

# Kernel config.
INPUT_SIZES=( 1024 2048 4096 8192 )
OUTPUT_SIZES=( 128 256 512 )
if [ ! "${DATA_TYPE}" ]; then
    DATA_TYPE="f32"
fi

echo "Result type: GFLOPS"
for OUT_SIZE in "${OUTPUT_SIZES[@]}"; do
  echo "MLP - OUT: ${OUT_SIZE} INS: ${INPUT_SIZES[@]}"
  for IN_SIZE in "${INPUT_SIZES[@]}"; do
    # Run benchmark.
    NUM_ITER=10000
    FUSE_TYPE=5
    TYPE=F
    TILES=(64 64 64)
    LAYOUT=(0 0)
    if [ "${DATA_TYPE}" = "bf16" ]; then
        LAYOUT=(1 1)
    fi
    # Disable parallelism.
    ENV_FLAGS=OMP_NUM_THREADS=1
    exec env ${ENV_FLAGS} ${BENCH_RUNNER} ${NUM_ITER} ${OUT_SIZE} ${FUSE_TYPE} ${TYPE} ${TILES[@]} \
        ${LAYOUT[@]} ${IN_SIZE} ${OUT_SIZE} \
        | sed -nE "s/.*GFLOPS\s+=\s*([0-9.]+).*/\\1/p"
  done
done
