// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

using CPUSpecificParams =  std::tuple<
    std::vector<cpu_memory_format_t>, // input memomry format
    std::vector<cpu_memory_format_t>, // output memory format
    std::vector<std::string>,         // priority
    std::string                       // selected primitive type
>;

namespace CPUTestUtils {
std::vector<CPUSpecificParams> filterCPUInfo(const std::vector<CPUSpecificParams>& CPUParams);
std::vector<CPUSpecificParams> filterCPUInfoForArch(const std::vector<CPUSpecificParams>& CPUParams);
std::vector<CPUSpecificParams> filterCPUInfoForDevice(const std::vector<CPUSpecificParams>& CPUParams);
} // namespace CPUTestUtils
