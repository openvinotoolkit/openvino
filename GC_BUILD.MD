### Tested on x1-spr 'Triton (GPU, Agama 2521.10, DLE 2025.1.1, Ubuntu 22.04)' profile

#### step 1: build llvm (tested on 2634a2bda1db92ab5324a47459ee7f23e531ce53)
Important! RTTI has to be enabled!
```
cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_DUMP=1 \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=true \
    -DLLVM_ENABLE_PROJECTS="mlir;lld" \
    -DLLVM_TARGETS_TO_BUILD="X86;SPIRV" \
    -DLLVM_INSTALL_UTILS=true \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DLLVM_ENABLE_RTTI=ON \
    -DLLVM_ENABLE_EH=ON \
    -DCMAKE_INSTALL_PREFIX=/home/jovyan/llvm/curr_install

cmake --build . --target install
```

#### step 2: install opencl-headers & install llvm-env-vars
```
sudo apt install -y intel-opencl-icd opencl-c-headers ocl-icd-opencl-dev
export LLVM_INST_PATH=/home/jovyan/llvm/build/
```

#### step 3: build graph-compiler

```
# clone
git clone https://github.com/intel-sandbox/graph-compiler.git
cd graph-compiler
mkdir build && cd build

# install nanobind
pip install nanobind

# build
cmake ../ -G Ninja \
	-DLLVM_DIR=$LLVM_INST_PATH/lib/cmake/llvm \
	-DMLIR_DIR=$LLVM_INST_PATH/lib/cmake/mlir \
    -DCMAKE_INSTALL_PREFIX=/home/jovyan/graph-compiler/install

cmake --build . --target install
```

#### step 4: build openvino

```
git clone https://github.com/intel-sandbox/openvino-gc
cd openvino
git checkout mlir-gc-integration

mkdir build && cd build
cmake ../ -G Ninja -DLLVM_DIR=$LLVM_INST_PATH/lib/cmake/llvm \
                   -DMLIR_DIR=$LLVM_INST_PATH/lib/cmake/mlir \
                   -DENABLE_GRAPH_COMPILER=ON \
                   -DENABLE_INTEL_GPU=ON \
                   -DENABLE_TESTS=ON \
		   -DENABLE_ONEDNN_FOR_GPU=OFF \
		   -DENABLE_INTEL_CPU=OFF \
                   -DCMAKE_CXX_FLAGS="-DOV_GPU_OPENCL_HPP_HAS_UUID -DOV_GPU_OPENCL_HPP_HAS_BUS_INFO" \
                   -DGraphCompiler_DIR=/home/jovyan/graph-compiler/install/lib/cmake/GraphCompiler
cmake --build . -j16
```

#### step 5: run sanity-mlir tests and dump mlir

```
OV_MLIR_DEBUG=1 OV_MLIR_MODE=GC ./bin/intel64/Release/ov_gpu_func_tests --gtest_filter=MLIRExecution.SimpleMatmulf32
```

### CI runner
The CI runner is the X1 server - gcrun. When the session starts, it installs and starts itself automatically. The action scripts are located in the graph-compiler repository.
