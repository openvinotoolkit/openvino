#!/bin/bash

echo "### MLP BF16 benchmarks ###"
echo "LIBXSMM"
../tools/mlir_bench/libxsmm_bench.sh -B
echo "TPP-MLIR args weights"
../tools/mlir_bench/tpp_mlir_bench.sh -t bf16

echo ""
echo "Baseline MLP"
echo "OV - no MLIR - baseline model"
OV_MLIR=0 ../tools/mlir_bench/mlp_bench.sh -t bf16 -b mlp
echo "OV + MLIR - kernel only - baseline model"
../tools/mlir_bench/ov_raw_mlir_bench.sh -t bf16 -b mlp
echo "OV + MLIR - full - baseline model"
OV_MLIR=1 ../tools/mlir_bench/mlp_bench.sh -t bf16 -b mlp

echo ""
echo "PyTorch MLP"
echo "OV - no MLIR - PyTorch"
OV_MLIR=0 ../tools/mlir_bench/mlp_bench.sh -t bf16
echo "OV + MLIR - kernel only - PyTorch"
../tools/mlir_bench/ov_raw_mlir_bench.sh -t bf16
echo "OV + MLIR - full - PyTorch"
OV_MLIR=1 ../tools/mlir_bench/mlp_bench.sh -t bf16

echo "TPP-MLIR const weights"
../tools/mlir_bench/tpp_mlir_bench.sh -t bf16 -C
