# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ov_option(ENABLE_MLIR_COMPILER "Enable compilation of npu_mlir_compiler libraries" ON)

ov_option(ENABLE_NPU_RUNTIME_COMMON "Enable compilation of npu runtime common libraries" ON)

# if ENABLE_ZEROAPI_BACKEND=ON, it adds the ze_loader dependency for driver compiler
ov_dependent_option(ENABLE_NPU_PLUGIN_ENGINE "Enable compilation of NPU plugin engine" ON "ENABLE_NPU_RUNTIME_COMMON" OFF)

ov_dependent_option(ENABLE_ZEROAPI_BACKEND "Enable zero-api as a plugin backend" ON "ENABLE_NPU_RUNTIME_COMMON;ENABLE_NPU_PLUGIN_ENGINE" OFF)

ov_dependent_option(ENABLE_DRIVER_COMPILER_ADAPTER "Enable NPU Compiler inside driver" ON "ENABLE_ZEROAPI_BACKEND" OFF)

if((NOT ENABLE_NPU_PLUGIN_ENGINE OR NOT ENABLE_NPU_RUNTIME_COMMON) AND ENABLE_TESTS)
    message(FATAL_ERROR "Tests depends on npu plugin engine and npu runtime common libraries!")
endif()

if((NOT ENABLE_NPU_PLUGIN_ENGINE OR NOT ENABLE_NPU_RUNTIME_COMMON) AND ENABLE_ZEROAPI_BACKEND)
    message(FATAL_ERROR "Zero backend depends on npu plugin engine and npu common libraries!")
endif()

if(NOT ENABLE_ZEROAPI_BACKEND AND ENABLE_DRIVER_COMPILER_ADAPTER)
    message(FATAL_ERROR "Compiler adapter depends on zero backend to use same context!")
endif()

if(NOT BUILD_SHARED_LIBS AND NOT ENABLE_MLIR_COMPILER AND NOT ENABLE_DRIVER_COMPILER_ADAPTER)
    message(FATAL_ERROR "No compiler found for static build!")
endif()

ov_dependent_option(ENABLE_IMD_BACKEND "Enable InferenceManagerDemo based NPU AL backend" OFF "NOT WIN32;NOT CMAKE_CROSSCOMPILING" OFF)

ov_dependent_option(ENABLE_INTEL_NPU_PROTOPIPE "Enable Intel NPU Protopipe tool" ON "ENABLE_INTEL_NPU_INTERNAL" OFF)
