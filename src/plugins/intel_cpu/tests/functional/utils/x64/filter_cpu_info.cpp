// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/cpu_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"

namespace CPUTestUtils {

std::vector<CPUSpecificParams> filterCPUInfo(const std::vector<CPUSpecificParams>& CPUParams) {
    std::vector<CPUSpecificParams> archCPUParams = filterCPUInfoForArch(CPUParams);
    std::vector<CPUSpecificParams> deviceCPUParams = filterCPUInfoForDevice(archCPUParams);
    return deviceCPUParams;
}

std::vector<CPUSpecificParams> filterCPUInfoForArch(const std::vector<CPUSpecificParams>& CPUParams) {
    std::vector<CPUSpecificParams> resCPUParams;
    const int selectedTypeIndex = 3;

    for (auto param : CPUParams) {
        auto selectedTypeStr = std::get<selectedTypeIndex>(param);

        if (selectedTypeStr.find("acl") != std::string::npos)
            continue;

        resCPUParams.push_back(param);
    }

    return resCPUParams;
}

std::vector<CPUSpecificParams> filterCPUInfoForDevice(const std::vector<CPUSpecificParams>& CPUParams) {
    std::vector<CPUSpecificParams> resCPUParams;
    const int selectedTypeIndex = 3;

    for (auto param : CPUParams) {
        auto selectedTypeStr = std::get<selectedTypeIndex>(param);

        if (selectedTypeStr.find("jit") != std::string::npos && !ov::with_cpu_x86_sse42())
            continue;
        if (selectedTypeStr.find("sse42") != std::string::npos && !ov::with_cpu_x86_sse42())
            continue;
        if (selectedTypeStr.find("avx") != std::string::npos && !ov::with_cpu_x86_avx())
            continue;
        if (selectedTypeStr.find("avx2") != std::string::npos && !ov::with_cpu_x86_avx2())
            continue;
        if (selectedTypeStr.find("avx512") != std::string::npos && !ov::with_cpu_x86_avx512f())
            continue;
        if (selectedTypeStr.find("amx") != std::string::npos && !ov::with_cpu_x86_avx512_core_amx())
            continue;

        resCPUParams.push_back(param);
    }

    return resCPUParams;
}

} // namespace CPUTestUtils
