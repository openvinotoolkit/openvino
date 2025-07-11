# ARM64 Cross-compilation toolchain for Clang/LLVM
# Usage: cmake -DCMAKE_TOOLCHAIN_FILE=cmake/arm64-clang.toolchain.cmake ..

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Set Clang compiler with full paths
find_program(CMAKE_C_COMPILER NAMES clang PATHS /usr/bin /usr/local/bin)
find_program(CMAKE_CXX_COMPILER NAMES clang++ PATHS /usr/bin /usr/local/bin)

if(NOT CMAKE_C_COMPILER)
    message(FATAL_ERROR "clang not found")
endif()

if(NOT CMAKE_CXX_COMPILER)
    message(FATAL_ERROR "clang++ not found")
endif()

# Set the target triple for ARM64
set(CMAKE_C_COMPILER_TARGET aarch64-linux-gnu)
set(CMAKE_CXX_COMPILER_TARGET aarch64-linux-gnu)

# Set LLVM tools with full paths
find_program(CMAKE_AR NAMES llvm-ar PATHS /usr/bin /usr/local/bin)
find_program(CMAKE_RANLIB NAMES llvm-ranlib PATHS /usr/bin /usr/local/bin)
find_program(CMAKE_NM NAMES llvm-nm PATHS /usr/bin /usr/local/bin)
find_program(CMAKE_OBJDUMP NAMES llvm-objdump PATHS /usr/bin /usr/local/bin)
find_program(CMAKE_STRIP NAMES llvm-strip PATHS /usr/bin /usr/local/bin)
find_program(CMAKE_ASM_COMPILER NAMES clang PATHS /usr/bin /usr/local/bin)

if(NOT CMAKE_AR)
    message(FATAL_ERROR "llvm-ar not found")
endif()

# Cross-compilation flags
set(CMAKE_C_FLAGS_INIT "-target aarch64-linux-gnu")
set(CMAKE_CXX_FLAGS_INIT "-target aarch64-linux-gnu")
set(CMAKE_ASM_FLAGS_INIT "-target aarch64-linux-gnu")

# Assembly specific flags for ARM64
set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS_INIT} -x assembler-with-cpp" CACHE STRING "" FORCE)
set(CMAKE_ASM_COMPILER_FLAGS "-target aarch64-linux-gnu")
set(CMAKE_ASM_COMPILER_ID "Clang")
set(CMAKE_ASM_COMPILER_ID_RUN TRUE)
set(CMAKE_ASM_COMPILER_FORCED TRUE)

# Use lld linker
set(CMAKE_EXE_LINKER_FLAGS_INIT "-fuse-ld=lld -target aarch64-linux-gnu")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "-fuse-ld=lld -target aarch64-linux-gnu")
set(CMAKE_MODULE_LINKER_FLAGS_INIT "-fuse-ld=lld -target aarch64-linux-gnu")

# Search paths for libraries and headers
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# No sysroot - build will use host libraries but target ARM64
unset(CMAKE_SYSROOT)
set(CMAKE_FIND_ROOT_PATH "")

# Disable incompatible features for cross-compilation
set(CMAKE_CROSSCOMPILING TRUE)
set(CMAKE_CROSSCOMPILING_EMULATOR "")

# Set assembler target for ARM64
set(CMAKE_ASM_COMPILER_TARGET aarch64-linux-gnu)

# Force shared libraries build
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries" FORCE)

# Architecture specific settings for ARM64
set(DNNL_TARGET_ARCH "AARCH64" CACHE STRING "Target architecture" FORCE)
set(OV_CPU_ARM_TARGET_ARCH "arm64-v8a" CACHE STRING "ARM target" FORCE)

# Cross-compilation specific settings
set(CMAKE_SYSTEM_PROCESSOR_MODE "aarch64" CACHE STRING "" FORCE)
set(ARM_COMPUTE_TARGET_ARCH "arm64-v8a" CACHE STRING "" FORCE)
set(ARM_COMPUTE_ARCH "arm64-v8a" CACHE STRING "" FORCE)

# Compiler flags for ARM64 assembly
set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS_INIT} -x assembler-with-cpp" CACHE STRING "" FORCE)

# CPU features for ARM64
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS_INIT} -march=armv8-a+fp+simd" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_INIT} -march=armv8-a+fp+simd" CACHE STRING "" FORCE)

