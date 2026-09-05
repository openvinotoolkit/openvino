include_guard()

set(SUPPORTED_LLVM_VERSION "24" CACHE STRING "")

find_package(LLVM CONFIG QUIET)
if (NOT LLVM_FOUND)
    set(LLVM_DIR "/usr/lib/llvm-${SUPPORTED_LLVM_VERSION}/lib/cmake/llvm" CACHE PATH "" FORCE)
    find_package(LLVM REQUIRED CONFIG)
endif()

find_package(MLIR CONFIG QUIET)
if (NOT MLIR_FOUND)
    get_filename_component(llvm_cmake_path "${LLVM_DIR}" REALPATH)
    get_filename_component(llvm_cmake_dir "${llvm_cmake_path}" DIRECTORY)
    set(MLIR_DIR "${llvm_cmake_dir}/mlir" CACHE PATH "" FORCE)
    find_package(MLIR REQUIRED CONFIG)
endif()

message(STATUS "LLVM_DIR: ${LLVM_DIR}")
message(STATUS "MLIR_DIR: ${MLIR_DIR}")
