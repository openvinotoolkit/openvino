include_guard()

find_package(LLVM CONFIG QUIET)
find_package(MLIR CONFIG QUIET)

if (NOT LLVM_FOUND OR NOT MLIR_FOUND)
    # Try apt.llvm.org install paths explicitly
    set(LLVM_VERSION "23" CACHE STRING "LLVM nightly major version from apt.llvm.org")
    set(llvm_apt_dir "/usr/lib/llvm-${LLVM_VERSION}/lib/cmake/llvm")
    set(mlir_apt_dir "/usr/lib/llvm-${LLVM_VERSION}/lib/cmake/mlir")

    if (EXISTS "${llvm_apt_dir}/LLVMConfig.cmake" AND EXISTS "${mlir_apt_dir}/MLIRConfig.cmake")
        set(LLVM_DIR "${llvm_apt_dir}" CACHE PATH "" FORCE)
        set(MLIR_DIR "${mlir_apt_dir}" CACHE PATH "" FORCE)
    else()
        message(FATAL_ERROR
            "LLVM/MLIR not found. Either install LLVM nightly:\n"
            "  sudo ./install_build_dependencies.sh -llvm\n"
            "Or add the following CMake options:\n"
            "  -DLLVM_DIR=path/to/llvm/lib/cmake/llvm\n"
            "  -DMLIR_DIR=path/to/llvm/lib/cmake/mlir\n"
        )
    endif()
endif()

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "LLVM ${LLVM_PACKAGE_VERSION} at ${LLVM_DIR}")
message(STATUS "MLIR at ${MLIR_DIR}")
