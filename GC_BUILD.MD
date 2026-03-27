### Tested on x1-spr 'Triton (GPU, Agama 2521.10, DLE 2025.1.1, Ubuntu 22.04)' profile

If you feel that these steps look outdated, check the actual CI-steps here: https://github.com/intel-sandbox/graph-compiler/blob/main/.github/workflows/ci.yml

#### step 1: install opencl-headers & install llvm-env-vars
```
sudo apt install -y intel-opencl-icd opencl-c-headers ocl-icd-opencl-dev
```

#### step 2: build graph-compiler + llvm

```
# clone
git clone https://github.com/intel-sandbox/graph-compiler.git
cd graph-compiler
# install nanobind
pip install nanobind

./scripts/compile.sh -l
```

#### step 4: build openvino

```
git clone https://github.com/intel-sandbox/openvino-gc
cd openvino
git checkout mlir-gc-integration

mkdir build && cd build
cmake -G Ninja -S "$ov_dir" -B "$ov_dir/build" \
            -DLLVM_DIR=$llvm_dir/lib/cmake/llvm \
            -DMLIR_DIR=$llvm_dir/lib/cmake/mlir \
            -DENABLE_GRAPH_COMPILER=ON \
            -DLLVM_DYLINK=ON \
            -DENABLE_INTEL_GPU=ON \
            -DENABLE_TESTS=ON \
            -DENABLE_ONEDNN_FOR_GPU=OFF \
            -DENABLE_INTEL_CPU=OFF \
            -DENABLE_INTEL_NPU=OFF \
            -DCMAKE_CXX_FLAGS="-DOV_GPU_OPENCL_HPP_HAS_UUID -DOV_GPU_OPENCL_HPP_HAS_BUS_INFO" \
            -DGraphCompiler_DIR=$gc_dir/lib/cmake/GraphCompiler \
            -DOpenCL_HPP_INCLUDE_DIR="$ov_dir/thirdparty/ocl/clhpp_headers/include" \
            -DOpenCL_HPP="$ov_dir/thirdparty/ocl/clhpp_headers/include/CL/opencl.hpp"
cmake --build "$ov_dir/build" -j16
```

#### step 5: run sanity-mlir tests and dump mlir

```
OV_MLIR_DEBUG=1 OV_MLIR_MODE=GC_GPU ./bin/intel64/Release/ov_gpu_func_tests --gtest_filter=MLIRExecution.SimpleMatmulf16
```

### CI runner
The CI runner is the X1 server - gcrun. When the session starts, it installs and starts itself automatically. The action scripts are located in the graph-compiler repository.
