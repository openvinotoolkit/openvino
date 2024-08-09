#!/bin/bash

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Runs MLIR only MLP benchmarks using TPP-MLIR.

die_syntax() {
  echo "Syntax: $0 [-t (f32|f16|bf16|...)] [-D] [-C] [-l 3]"
  echo ""
  echo "  -t: Optional data type"
  echo "  -l: Optional number of layers (def:3)"
  echo "  -D: Set model shapes to dynamic"
  echo "  -C: Weights as constants (default: arguments)"
  exit 1
}

# Cmd-line opts
while getopts "t:l:DC" arg; do
  case ${arg} in
    t)
      DATA_TYPE=${OPTARG}
      ;;
    l)
      NUM_LAYERS=${OPTARG}
      ;;
    D)
      IS_DYNAMIC=true
      ;;
    C)
      CONST_WEIGHTS=true
      ;;
    ?)
      echo "Invalid option: ${OPTARG}"
      die_syntax
      ;;
  esac
done

MODEL_GEN=mlir-gen
BENCH_RUNNER=tpp-run

# Initial validation.
if ! [ "$(command -v ${MODEL_GEN})" ]; then
  echo "Missing model generator ${MODEL_GEN}"
  exit 1
fi
if ! [ "$(command -v ${BENCH_RUNNER})" ]; then
  echo "Missing benchmark runner ${BENCH_RUNNER}"
  exit 1
fi
if [ "${IS_DYNAMIC}" ]; then
  echo "Dynamic shapes are not supported by ${BENCH_RUNNER}"
  exit 1
fi

# Kernel config.
# LAYERS=( 1024 2048 4096 8192 )
# MINI_BATCHES=( 128 256 512 )
LAYERS=( 1024 )
MINI_BATCHES=( 256 )
if [ ! "${DATA_TYPE}" ]; then
    DATA_TYPE="f32"
fi
if [ ! $NUM_LAYERS ]; then
  NUM_LAYERS=3
fi

echo "Result type: time [s] - NUM LAYERS: ${NUM_LAYERS}"
for MB in "${MINI_BATCHES[@]}"; do
  echo "MLP - MB: ${MB} LAYERS: ${LAYERS[@]}"
  for LAYER in "${LAYERS[@]}"; do
    # Generate model.
    LAYER_STRING="${LAYER}"
    for i in $(seq ${NUM_LAYERS}); do
      LAYER_STRING="${LAYER_STRING},${LAYER}"
    done
    MODEL_CONFIG=(--batch=${MB} --layers=${LAYER_STRING} -bias -relu)
    KERNEL_TYPE=args
    if [ "${CONST_WEIGHTS}" ]; then
        KERNEL_TYPE=const
    fi
    GEN_FLAGS=(--kernel=${KERNEL_TYPE} --float-type=${DATA_TYPE} --seed=123)
    MLIR_IR=$(${MODEL_GEN} "${MODEL_CONFIG[@]}" "${GEN_FLAGS[@]}")
    if [ $? != 0 ]; then
        echo "Failed to generate model"
        exit 1
    fi
    # Run benchmark.
    BENCH_FLAGS="-entry-point-result=void -e entry -seed 123 -n 1000"
    echo "${MLIR_IR}" | ${BENCH_RUNNER} ${BENCH_FLAGS}
  done
done
