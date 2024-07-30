#!/bin/bash

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Runs pure MLIR part of MLP benchmarks using TPP-MLIR.
# This approach assumes that only one MLIR op is generated.
# For example, the whole graph is outlined to MLIR.

die_syntax() {
  echo "Syntax: $0 [-t (f32|f16|bf16|...)] [-b (mlp)] [-D]"
  echo ""
  echo "  -t: Optional data type"
  echo "  -b: Optional baseline model"
  echo "  -D: Set model shapes to dynamic"
  exit 1
}

# Cmd-line opts
while getopts "t:b:D" arg; do
  case ${arg} in
    t)
      DATA_TYPE=${OPTARG}
      ;;
    b)
      BASELINE_MODEL=${OPTARG}
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

OV_ROOT=$(git rev-parse --show-toplevel)
BENCH_ROOT=$(realpath ${OV_ROOT}/tools/mlir_bench)

MODEL_GEN=$(realpath ${BENCH_ROOT}/ov_model_gen.py)
BENCH_RUNNER=tpp-run

# Initial validation.
if ! [ -d ${OV_ROOT} ]; then
  echo "Missing OV repo"
  exit 1
fi
if ! [ -d ${BENCH_ROOT} ]; then
  echo "Missing MLIR benchmark directory"
  exit 1
fi
if ! [ -f ${MODEL_GEN} ]; then
  echo "Missing model generator"
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
MODEL_NAME="TPP_BENCH.xml"

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
    GEN_FLAGS=(-t ${DATA_TYPE} -n ${MODEL_NAME})
    GEN_FLAGS+=(-p)
    ENV_FLAGS=OV_MLIR_TPP=0
    MODEL_OUT=$(exec env ${ENV_FLAGS} python3 ${MODEL_GEN} "${MODEL_CONFIG[@]}" "${GEN_FLAGS[@]}" 2>&1)
    if [ $? != 0 ]; then
        echo "Failed to generate model"
        exit 1
    fi
    # Run benchmark.
    MLIR_IR=$(echo "${MODEL_OUT}" \
        | awk '/Source MLIR:/{flag=1; next} /Target LLVM:/{flag=0} flag' \
        | grep -vE '^[-]+$')
    BENCH_FLAGS="-entry-point-result=void -e entry -seed 123 -n 10000"
    echo "${MLIR_IR}" | ${BENCH_RUNNER} ${BENCH_FLAGS}
  done
done
