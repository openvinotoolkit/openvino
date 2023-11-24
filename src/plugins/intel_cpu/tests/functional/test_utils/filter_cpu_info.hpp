// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_test_utils.hpp"
#include <vector>

namespace CPUTestUtils {
std::vector<CPUSpecificParams> filterCPUInfo(const std::vector<CPUSpecificParams>& CPUParams);
std::vector<CPUSpecificParams> filterCPUInfoForArch(const std::vector<CPUSpecificParams>& CPUParams);
std::vector<CPUSpecificParams> filterCPUInfoForDevice(const std::vector<CPUSpecificParams>& CPUParams);
// TODO Split test case of X86_64 and RISCV64
#if defined(OPENVINO_ARCH_X86_64) || defined(OPENVINO_ARCH_RISCV64)
std::vector<CPUSpecificParams> filterCPUInfoForDeviceWithFP16(const std::vector<CPUSpecificParams>& allParams);
#endif
} // namespace CPUTestUtils
