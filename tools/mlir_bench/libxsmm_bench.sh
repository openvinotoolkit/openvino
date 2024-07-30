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
if [ "${IS_DYNAMIC}" ]; then
  echo "Dynamic shapes are not supported by ${BENCH_RUNNER}"
  exit 1
fi

# Kernel config.
LAYERS=( 1024 2048 4096 8192 )
MINI_BATCHES=( 128 256 512 )
if [ ! "${DATA_TYPE}" ]; then
    DATA_TYPE="f32"
fi

echo "Result type: GFLOPS"
for MB in "${MINI_BATCHES[@]}"; do
  echo "MLP - MB: ${MB} LAYERS: ${LAYERS[@]}"
  for LAYER in "${LAYERS[@]}"; do
    # Run benchmark.
    NUM_ITER=1000
    FUSE_TYPE=5
    TYPE=F
    TILES=(64 64 64)
    LAYOUT=(0 0)
    if [ "${DATA_TYPE}" = "bf16" ]; then
        LAYOUT=(1 1)
    fi
    # Disable parallelism.
    ENV_FLAGS=OMP_NUM_THREADS=1
    exec env ${ENV_FLAGS} ${BENCH_RUNNER} ${NUM_ITER} ${MB} ${FUSE_TYPE} ${TYPE} ${TILES[@]} \
        ${LAYOUT[@]} ${LAYER} ${LAYER} \
        | sed -nE "s/.*GFLOPS\s+=\s*([0-9.]+).*/\\1/p"
  done
done
