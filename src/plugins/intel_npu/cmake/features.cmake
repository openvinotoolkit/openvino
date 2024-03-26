# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ov_option(ENABLE_MLIR_COMPILER "Enable compilation of npu_mlir_compiler libraries" ON)

ov_option(BUILD_COMPILER_FOR_DRIVER "Enable build of npu_driver_compiler" OFF)

ov_dependent_option(ENABLE_DRIVER_COMPILER_ADAPTER "Enable NPU Compiler inside driver" ON "NOT BUILD_COMPILER_FOR_DRIVER" OFF)

if(NOT BUILD_SHARED_LIBS AND NOT ENABLE_MLIR_COMPILER AND NOT ENABLE_DRIVER_COMPILER_ADAPTER)
    message(FATAL_ERROR "No compiler found for static build!")
endif()

# if ENABLE_ZEROAPI_BACKEND=ON, it adds the ze_loader dependency for driver compiler
ov_dependent_option(ENABLE_ZEROAPI_BACKEND "Enable zero-api as a plugin backend" ON "NOT BUILD_COMPILER_FOR_DRIVER" OFF)

ov_dependent_option(ENABLE_IMD_BACKEND "Enable InferenceManagerDemo based NPU AL backend" OFF "NOT WIN32;NOT CMAKE_CROSSCOMPILING" OFF)
