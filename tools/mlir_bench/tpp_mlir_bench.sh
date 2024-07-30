#!/bin/bash

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Runs MLIR only MLP benchmarks using TPP-MLIR.

die_syntax() {
  echo "Syntax: $0 [-t (f32|f16|bf16|...)] [-D]"
  echo ""
  echo "  -t: Optional data type"
  echo "  -D: Set model shapes to dynamic"
  exit 1
}

# Cmd-line opts
while getopts "t:D" arg; do
  case ${arg} in
    t)
      DATA_TYPE=${OPTARG}
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

echo "Result type: time [ns]"
for OUT_SIZE in "${OUTPUT_SIZES[@]}"; do
  echo "MLP - OUT: ${OUT_SIZE} INS: ${INPUT_SIZES[@]}"
  for IN_SIZE in "${INPUT_SIZES[@]}"; do
    # Generate model.
    if [ "${BASELINE_MODEL}" ]; then
        # Enable baseline model flag.
        MODEL_CONFIG=(-b="${BASELINE_MODEL}[${OUT_SIZE},${OUT_SIZE},${IN_SIZE}]")
    else
        # Generate default PyTorch MLP.
        MODEL_CONFIG=(-l="linear[${IN_SIZE},${OUT_SIZE}] relu[]")
    fi
    MODEL_CONFIG=(--batch=${OUT_SIZE} --layers=${IN_SIZE},${OUT_SIZE} -bias -relu)
    GEN_FLAGS=(--kernel=args --float-type=${DATA_TYPE} --seed=123)
    MLIR_IR=$(${MODEL_GEN} "${MODEL_CONFIG[@]}" "${GEN_FLAGS[@]}")
    if [ $? != 0 ]; then
        echo "Failed to generate model"
        exit 1
    fi
    # Run benchmark.
    BENCH_FLAGS="-entry-point-result=void -e entry -seed 123 -n 10000"
    echo "${MLIR_IR}" | ${BENCH_RUNNER} ${BENCH_FLAGS}
  done
done
