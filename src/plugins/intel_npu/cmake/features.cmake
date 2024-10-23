# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ov_option(ENABLE_MLIR_COMPILER "Enable compilation of npu_mlir_compiler libraries" ON)

ov_option(ENABLE_NPU_RUNTIME_COMMON "Enable compilation of npu runtime common libraries" ON)

ov_dependent_option(ENABLE_NPU_PLUGIN_ENGINE "Enable compilation of NPU plugin engine" ON "ENABLE_NPU_RUNTIME_COMMON" OFF)

if((NOT ENABLE_NPU_PLUGIN_ENGINE OR NOT ENABLE_NPU_RUNTIME_COMMON) AND ENABLE_TESTS)
    message(FATAL_ERROR "Tests depends on npu plugin engine and npu runtime common libraries!")
endif()

ov_dependent_option(ENABLE_IMD_BACKEND "Enable InferenceManagerDemo based NPU AL backend" OFF "NOT WIN32;NOT CMAKE_CROSSCOMPILING" OFF)

ov_dependent_option(ENABLE_INTEL_NPU_PROTOPIPE "Enable Intel NPU Protopipe tool" ON "ENABLE_INTEL_NPU_INTERNAL" OFF)
