#!/bin/bash

die_syntax() {
  echo "Syntax: $0 [-l 3]"
  echo ""
  echo "  -l: Optional number of layers (def: 3)"
  exit 1
}

# Cmd-line opts
while getopts "l:" arg; do
  case ${arg} in
    l)
      NUM_LAYERS=${OPTARG}
      ;;
    ?)
      echo "Invalid option: ${OPTARG}"
      die_syntax
      ;;
  esac
done

if [ ! "${NUM_LAYERS}" ]; then
  NUM_LAYERS=3
fi

export OV_MLIR_DEBUG=1

echo "### MLP BF16 benchmarks ###"
echo "# Layers: ${NUM_LAYERS} #"
echo "LIBXSMM"
../tools/mlir_bench/libxsmm_bench.sh -B -l ${NUM_LAYERS}
echo "TPP-MLIR args weights"
../tools/mlir_bench/tpp_mlir_bench.sh -t bf16 -l ${NUM_LAYERS}

echo ""
echo "Baseline MLP"
echo "OV - no MLIR - baseline model"
OV_MLIR=0 ../tools/mlir_bench/mlp_bench.sh -t bf16 -b mlp -l ${NUM_LAYERS}
echo "OV + MLIR - kernel only - baseline model"
../tools/mlir_bench/ov_raw_mlir_bench.sh -t bf16 -b mlp -l ${NUM_LAYERS}
echo "OV + MLIR - full - baseline model"
OV_MLIR=1 ../tools/mlir_bench/mlp_bench.sh -t bf16 -b mlp -l ${NUM_LAYERS}

echo ""
echo "PyTorch MLP"
echo "OV - no MLIR - PyTorch"
OV_MLIR=0 ../tools/mlir_bench/mlp_bench.sh -t bf16 -l ${NUM_LAYERS}
echo "OV + MLIR - kernel only - PyTorch"
../tools/mlir_bench/ov_raw_mlir_bench.sh -t bf16 -l ${NUM_LAYERS}
echo "OV + MLIR - full - PyTorch"
OV_MLIR=1 ../tools/mlir_bench/mlp_bench.sh -t bf16 -l ${NUM_LAYERS}

echo "TPP-MLIR const weights"
../tools/mlir_bench/tpp_mlir_bench.sh -t bf16 -C -l ${NUM_LAYERS}
