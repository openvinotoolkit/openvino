#!/bin/bash

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Runs OV MLP benchmarks.

die_syntax() {
  echo "Syntax: $0 [-t (f32|f16|bf16|...)] [-b (mlp)] [-D] [-l 3]"
  echo ""
  echo "  -t: Optional data type"
  echo "  -b: Optional baseline model"
  echo "  -l: Optional number of layers (def:3)"
  echo "  -D: Set model shapes to dynamic"
  exit 1
}

# Cmd-line opts
while getopts "t:l:b:D" arg; do
  case ${arg} in
    t)
      DATA_TYPE=${OPTARG}
      ;;
    b)
      BASELINE_MODEL=${OPTARG}
      ;;
    l)
      NUM_LAYERS=${OPTARG}
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

if [ ! $NUM_LAYERS ]; then
  NUM_LAYERS=3
fi

OV_ROOT=$(git rev-parse --show-toplevel)
BENCH_ROOT=$(realpath "${OV_ROOT}/tools/mlir_bench")

MODEL_GEN=$(realpath "${BENCH_ROOT}/ov_model_gen.py")
BENCH_RUNNER=benchmark_app

# Initial validation.
if ! [ -d "${OV_ROOT}" ]; then
  echo "Missing OV repo"
  exit 1
fi
if ! [ -d "${BENCH_ROOT}" ]; then
  echo "Missing MLIR benchmark directory"
  exit 1
fi
if ! [ -f "${MODEL_GEN}" ]; then
  echo "Missing model generator"
  exit 1
fi
if ! [ "$(command -v ${BENCH_RUNNER})" ]; then
  echo "Missing benchmark runner ${BENCH_RUNNER}"
  exit 1
fi
if [ "${BASELINE_MODEL}" ] && [ "${IS_DYNAMIC}" ]; then
  echo "Baseline models with dynamic shapes not supported"
  exit 1
fi

# Kernel config.
#LAYERS=( 1024 2048 4096 8192 )
#MINI_BATCHES=( 128 256 512 )
LAYERS=( 1024 )
MINI_BATCHES=( 256 )
if [ ! "${DATA_TYPE}" ]; then
    DATA_TYPE="f32"
fi
MODEL_NAME="MLIR_MLP_BENCH.xml"

echo "Result type: time [ms] - NUM LAYERS: ${NUM_LAYERS}"
for MB in "${MINI_BATCHES[@]}"; do
  echo "MLP - MB: ${MB} LAYERS: ${LAYERS[@]}"
  for LAYER in "${LAYERS[@]}"; do
    # Generate model.
    if [ "${BASELINE_MODEL}" ]; then
        # Enable baseline model flag.
        MODEL_CONFIG=(-b="${BASELINE_MODEL}[${MB},${LAYER},${LAYER}]x${NUM_LAYERS}")
    else
        # Generate default PyTorch MLP.
        LAYER_STRING="linear[${MB},${LAYER},${LAYER}] relu[]"
        for i in $(seq ${NUM_LAYERS}); do
          MODEL_STRING="${MODEL_STRING}${LAYER_STRING} "
        done
        MODEL_CONFIG=(-l="${MODEL_STRING}")
    fi
    echo "MODEL_CONFIG=${MODEL_CONFIG}"
    GEN_FLAGS=(-t ${DATA_TYPE} -n ${MODEL_NAME})
    if [ "${IS_DYNAMIC}" ]; then
        GEN_FLAGS+=(--dynamic)
    fi
    python3 ${MODEL_GEN} "${MODEL_CONFIG[@]}" "${GEN_FLAGS[@]}"
    if [ $? != 0 ]; then
        echo "Failed to generate model"
        exit 1
    fi
    # Run benchmark.
    PRECISION=${DATA_TYPE}
    if [ "${DATA_TYPE}" = "bf16" ]; then
        # No native support for bf16, use simple f16 instead.
        PRECISION="f16"
    fi
    if [ "${IS_DYNAMIC}" ]; then
        DATA_SHAPE=(-data_shape [${MB},${LAYER}])
    fi
    # Benchmark config. Disable parallelism.
    PERF_FLAGS="-niter 1000 -hint none -nstreams 1 -nthreads 1"
    BENCH_FLAGS="-m ${MODEL_NAME} -d CPU -ip ${PRECISION} -infer_precision ${DATA_TYPE} ${DATA_SHAPE[@]} ${PERF_FLAGS}"
    ${BENCH_RUNNER} ${BENCH_FLAGS} 2>/dev/null | \
        sed -nE "s/.*\[ INFO \]\s*Median:\s*([0-9.]+).*/\\1/p"
  done
done
