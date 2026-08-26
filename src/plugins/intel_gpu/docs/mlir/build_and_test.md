# How to build and test OV with MLIR support

MLIR and Graph Compiler are not added as third-party submodules to the project, so a developer has to manually
build a suitable LLVM & Graph Compiler and then provide cmake-configs during the OpenVINO build.

The pinned Graph Compiler revision can be found in [`cmake/graph-compiler.cmake`](../../../../../cmake/graph-compiler.cmake).

## 1. Clone a suitable graph-compiler and build gc + llvm

```sh
git clone --depth 1 --branch ov_pin/0.1.1 https://github.com/dchigarev/graph-compiler.git
cd graph-compiler
export GRAPH_COMPILER_DIR=$(pwd)

# This script clones a suitable llvm to `graph-compiler/externals/llvm-project`;
# builds llvm; builds graph-compiler
./scripts/compile.sh -r
```

## 2. Build OpenVINO with MLIR support

```sh
export GC_INSTALL_DIR=$GRAPH_COMPILER_DIR/build/install
export LLVM_INSTALL_DIR=$GRAPH_COMPILER_DIR/externals/llvm-project/build

cmake -S <openvino> -B <build> \
  -DENABLE_INTEL_GPU=ON \
  -DENABLE_GPU_MLIR=ON \
  -DGraphCompiler_DIR="${GC_INSTALL_DIR}/lib/cmake/GraphCompiler" \
  -DMLIR_DIR="${LLVM_INSTALL_DIR}/lib/cmake/mlir" \
  -DLLVM_DIR="${LLVM_INSTALL_DIR}/lib/cmake/llvm" \
  ...
```

## 3. Run

```sh
export OV_GPU_ENABLE_MLIR=1        # or GPU_ENABLE_MLIR in the GPU config file
```

## Testing

```
OV_GPU_ENABLE_MLIR=1 ./bin/intel64/Release/ov_gpu_func_tests --gtest_filter="*mlir*"
```

The MLIR path is tested via its own test suites located at `tests/functional/mlir_op` (the suites are **not**
included to the regular build, they are only built with `ENABLE_GPU_MLIR=ON`). The suites are mostly
copied from the usual OV tests (e.g. `functional/single_layer_tests/dynamic/scaled_dot_product_attention.cpp`
-> `functional/mlir_op/sdpa.cpp`) but redefine the tests base class (to `MlirSubgraphTest`), sometimes include
additional cases or change the accuracy thresholds that are suitable for the mlir implementations.

## See also

 * [MLIR execution path in GPU plugin](./overall_flow.md)
 * [OpenVINO GPU Plugin](../../README.md)
