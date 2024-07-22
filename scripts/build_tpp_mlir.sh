#!/bin/bash

# Run it in tpp-mlir/build subdirectory
# Set CUSTOM_LLVM_ROOT to llvm-project/build directory

cmake -G Ninja .. \
   -DCMAKE_BUILD_TYPE=RelWithDebInfo \
   -DMLIR_DIR=$CUSTOM_LLVM_ROOT/lib/cmake/mlir \
   -DLLVM_EXTERNAL_LIT=$CUSTOM_LLVM_ROOT/bin/llvm-lit \
   -DCMAKE_C_COMPILER=clang \
   -DCMAKE_CXX_COMPILER=clang++ \
   -DLLVM_USE_LINKER=lld

cmake --build . --target check-tpp