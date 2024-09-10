#!/usr/bin/env xonsh

# xonsh can be installed with `pip install xonsh`
# xonsh can then be run by invoking `python -m xonsh`
# this script in particular can be invoked with `python -m xonsh lora-benchmark.xsh`

import openvino as ov
from openvino.runtime.op import Constant
from openvino_devtools.builder import OpFactory, outputs_to_nodes
import numpy as np
from pprint import pprint
import re
from os import environ


CONFIGS = [
  [8], [16], [32], [64], [128], [256], [512], [1024]
]
ITERATIONS = 100

BENCH_RUNNER="tpp-run"
RUNNER_FLAGS=f"-entry-point-result=void -e entry -seed 123 -n {ITERATIONS}".split()
DEBUG = environ.get("OV_MLIR_DEBUG", "0").lower() in ("true", "1", "on")


def build_ov_lora_model(input_dim=-1, weight_dim=2048, lora_dim=8):
    opset = OpFactory('opset13')

    #t40 = opset.Parameter({'shape': [-1, -1, 2048], 'element_type': 'f32'}, output_names=[{'x'}])  # Input data
    t40 = opset.Parameter({'shape': [input_dim, weight_dim], 'element_type': 'f32'}, output_names=[{'x'}])  # Input data
    t52 = opset.Parameter({'shape': [1, lora_dim], 'element_type': 'f32'}, output_names=[{'alpha'}])  # LoRA alpha parameter

    t48 = Constant(np.random.rand(weight_dim, weight_dim).astype(np.float32))   #  -> f32[2048,2048]  # Original weight matrix W (usually it is compressed to bf16/f16/u8/u4 and represented as a sub-graph)
    t50 = Constant(np.random.rand(lora_dim, weight_dim).astype(np.float32))  #  -> f32[8,2048]   # LoRA matrix A
    t54 = Constant(np.random.rand(weight_dim, lora_dim).astype(np.float32))  #  -> f32[2048,8]   # LoRA matrix B

    t49 = opset.MatMul([t40, t48], {'transpose_a': False, 'transpose_b': True})  # f32[?,?,2048], f32[2048,2048] -> f32[?,?,2048]
    t51 = opset.MatMul([t40, t50], {'transpose_a': False, 'transpose_b': True})  # f32[?,?,2048], f32[8,2048] -> f32[?,?,8]
    t53 = opset.Multiply([t51, t52], {'auto_broadcast': 'numpy'})  # f32[?,?,8], f32[1,8] -> f32[?,?,8]
    t55 = opset.MatMul([t53, t54], {'transpose_a': False, 'transpose_b': True})  # f32[?,?,8], f32[2048,8] -> f32[?,?,2048]
    t56 = opset.Add([t49, t55], {'auto_broadcast': 'numpy'})  # f32[?,?,2048], f32[?,?,2048] -> f32[?,?,2048]
    t57 = opset.Result([t56], {})  # f32[?,?,2048] -> f32[?,?,2048]

    parameters = [t40, t52]
    results = [t57]
    sinks = []
    return ov.Model(outputs_to_nodes(results), outputs_to_nodes(sinks), outputs_to_nodes(parameters))


def build_mlir_lora_model(input_dim=-1, weight_dim=2048, lora_dim=8):
    input_dim = '?' if input_dim == -1 else input_dim
    mlir_model = f"\
!inputType = tensor<{input_dim}x{weight_dim}xf32>\n\
!loraAlphaType = tensor<1x{lora_dim}xf32>\n\
!weightType = tensor<{weight_dim}x{weight_dim}xf32>\n\
!loraMatAType = tensor<{lora_dim}x{weight_dim}xf32>\n\
!loraMatBType = tensor<{weight_dim}x{lora_dim}xf32>\n\
!loraResultType = tensor<{input_dim}x{lora_dim}xf32>\n\
func.func @entry(%arg0: !loraAlphaType, %arg1: !inputType) -> !inputType {{\n\
  %cst = arith.constant 0.000000e+00 : f32\n\
  %weights = arith.constant dense<0.001000e+00> : !weightType\n\
  %loraA = arith.constant dense<0.002000e+00> : !loraMatAType\n\
  %loraB = arith.constant dense<0.003000e+00> : !loraMatBType\n\
  %0 = tensor.empty() : !loraResultType\n\
  %1 = linalg.fill ins(%cst : f32) outs(%0 : !loraResultType)\
    -> !loraResultType\n\
  %2 = linalg.matmul_transpose_b ins(%arg1, %loraA : !inputType, !loraMatAType)\
    outs(%1 : !loraResultType) -> !loraResultType\n\
  %collapsed = tensor.collapse_shape %arg0 [[0, 1]] : !loraAlphaType into tensor<{lora_dim}xf32>\n\
  %broadcasted = linalg.broadcast ins(%collapsed : tensor<{lora_dim}xf32>)\
    outs(%0 : !loraResultType) dimensions = [0]\n\
  %3 = linalg.mul ins(%2, %broadcasted : !loraResultType, !loraResultType)\
    outs(%0 : !loraResultType) -> !loraResultType\n\
  %4 = tensor.empty() : !inputType\n\
  %5 = linalg.fill ins(%cst : f32) outs(%4 : !inputType) -> !inputType\n\
  %6 = linalg.matmul_transpose_b ins(%3, %loraB : !loraResultType, !loraMatBType)\
    outs(%5 : !inputType) -> !inputType\n\
  %7 = linalg.matmul_transpose_b ins(%arg1, %weights : !inputType, !weightType)\
    outs(%5 : !inputType) -> !inputType\n\
  %8 = linalg.add ins(%7, %6 : !inputType, !inputType) outs(%4 : !inputType) -> !inputType\n\
  return %8 : !inputType\n\
}}\n\
"
    return mlir_model


def main():
    no_mlir_averages = []
    mlir_averages = []
    no_ov_averages = []
    manual_mlir_averages = []
    for config in CONFIGS:
        model_desc = '.'.join(str(x) for x in config)
        model_xml = f"lora.{model_desc}.xml"
        model = build_ov_lora_model(*config)
        ov.save_model(model, model_xml)

        BENCH_FLAGS=f"-m {model_xml} -d CPU -ip f32 -infer_precision f32 -hint none -nstreams 1 -nthreads 1".split()

        def run_ov(env_str):
            out = $(env @(env_str.split()) benchmark_app @(BENCH_FLAGS) -niter @(ITERATIONS))
            match = re.search(r"Median: +(\d.*) ms", out)
            return float(match.group(1))
        no_mlir_averages.append(run_ov("OV_MLIR=0"))
        mlir_averages.append(run_ov("OV_MLIR=1"))

        def run_no_ov_mlir(env_str):
            raw_kernel_secs = $(env @(env_str.split()) benchmark_app @(BENCH_FLAGS) -niter 1 2>&1 | awk '/Source MLIR:/{flag=1; next} /Target LLVM:/{flag=0} flag' | grep -vE '^[-]+$' | tpp-run @(RUNNER_FLAGS))
            return float(raw_kernel_secs) * 1000
        no_ov_averages.append(run_no_ov_mlir("OV_MLIR=1 OV_MLIR_TPP=1 OV_MLIR_DEBUG=1"))

        def run_manual_mlir(env_str):
            mlir_model = build_mlir_lora_model(*config)
            if DEBUG:
              print(mlir_model)
            raw_kernel_secs = $(echo @(mlir_model) | tpp-run @(RUNNER_FLAGS))
            return float(raw_kernel_secs) * 1000
        manual_mlir_averages.append(run_manual_mlir(""))

    print("CONFIGS", CONFIGS)
    print("OV NO-MLIR", no_mlir_averages)
    print("OV MLIR", mlir_averages)
    print("NO-OV MLIR", no_ov_averages)
    print("MANUAL MLIR", manual_mlir_averages)


if __name__ == "__main__":
    main()
