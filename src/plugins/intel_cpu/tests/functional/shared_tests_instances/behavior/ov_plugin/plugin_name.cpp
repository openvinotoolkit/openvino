// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/visibility.hpp"

#if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
const char * cpu_plugin_file_name = "openvino_arm_cpu_plugin";
#elif defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
const char * cpu_plugin_file_name = "openvino_intel_cpu_plugin";
#elif defined(OPENVINO_ARCH_RISCV64)
const char * cpu_plugin_file_name = "openvino_riscv_cpu_plugin";
#else
#error "Undefined system processor"
#endif
