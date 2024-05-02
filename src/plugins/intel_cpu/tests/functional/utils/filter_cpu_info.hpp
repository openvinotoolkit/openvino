// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "cpu_test_utils.hpp"

namespace CPUTestUtils {
std::vector<CPUSpecificParams> filterCPUInfo(const std::vector<CPUSpecificParams>& CPUParams);
std::vector<CPUSpecificParams> filterCPUInfoForArch(const std::vector<CPUSpecificParams>& CPUParams);
std::vector<CPUSpecificParams> filterCPUInfoForDevice(const std::vector<CPUSpecificParams>& CPUParams);
}  // namespace CPUTestUtils
