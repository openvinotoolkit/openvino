# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ov_option(ENABLE_NPU_PLUGIN_ENGINE "Enable compilation of NPU plugin engine" ON)

ov_dependent_option(ENABLE_INTEL_NPU_PROTOPIPE "Enable Intel NPU Protopipe tool" ON "ENABLE_INTEL_NPU_INTERNAL" OFF)

ov_option(ENABLE_INTEL_NPU_COMPILER "Enable download of Intel NPU Compiler library" ${BUILD_SHARED_LIBS})
