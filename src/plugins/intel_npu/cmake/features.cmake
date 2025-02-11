# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ov_option(ENABLE_MLIR_COMPILER "Enable compilation of npu_mlir_compiler libraries" ON)

ov_option(ENABLE_NPU_PLUGIN_ENGINE "Enable compilation of NPU plugin engine" ON)

if(NOT ENABLE_NPU_PLUGIN_ENGINE AND ENABLE_TESTS)
    message(FATAL_ERROR "Tests depends on npu plugin engine!")
endif()

ov_dependent_option(ENABLE_IMD_BACKEND "Enable InferenceManagerDemo based NPU AL backend" OFF "NOT WIN32;NOT CMAKE_CROSSCOMPILING" OFF)

ov_dependent_option(ENABLE_INTEL_NPU_PROTOPIPE "Enable Intel NPU Protopipe tool" ON "ENABLE_INTEL_NPU_INTERNAL" OFF)
