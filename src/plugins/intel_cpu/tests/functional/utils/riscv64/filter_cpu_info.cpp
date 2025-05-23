// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/cpu_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"
#include "utils/general_utils.h"

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

        if (selectedTypeStr.find("shl") == std::string::npos &&
            selectedTypeStr.find("ref") == std::string::npos)
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

        if (selectedTypeStr.find("jit") != std::string::npos)
            continue;
        if (selectedTypeStr.find("acl") != std::string::npos)
            continue;
        if (selectedTypeStr.find("sse42") != std::string::npos)
            continue;
        if (selectedTypeStr.find("avx") != std::string::npos)
            continue;
        if (selectedTypeStr.find("avx2") != std::string::npos)
            continue;
        if (selectedTypeStr.find("avx512") != std::string::npos)
            continue;
        if (selectedTypeStr.find("amx") != std::string::npos)
            continue;

        resCPUParams.push_back(param);
    }
    return resCPUParams;
}

std::vector<CPUSpecificParams> filterCPUInfoForDeviceWithFP16(const std::vector<CPUSpecificParams>& allParams) {
    std::vector<CPUSpecificParams> resCPUParams;
    return resCPUParams;
}

std::vector<CPUSpecificParams> filterCPUSpecificParams(const std::vector<CPUSpecificParams> &paramsVector) {
    static const std::vector<CPUTestUtils::cpu_memory_format_t> supported_f = {CPUTestUtils::cpu_memory_format_t::nwc,
                                                                               CPUTestUtils::cpu_memory_format_t::ncw,
                                                                               CPUTestUtils::cpu_memory_format_t::nchw,
                                                                               CPUTestUtils::cpu_memory_format_t::nhwc,
                                                                               CPUTestUtils::cpu_memory_format_t::ndhwc,
                                                                               CPUTestUtils::cpu_memory_format_t::ncdhw};
    std::vector<CPUSpecificParams> filteredParamsVector = paramsVector;
    filteredParamsVector.erase(std::remove_if(filteredParamsVector.begin(),
                               filteredParamsVector.end(),
                               [](CPUSpecificParams param) {
                                                    const int inMemoryFormatTypeIndex = 0;
                                                    std::vector<CPUTestUtils::cpu_memory_format_t> inFormat = std::get<inMemoryFormatTypeIndex>(param);
                                                    const int outMemoryFormatIndex = 1;
                                                    std::vector<CPUTestUtils::cpu_memory_format_t> outFormat = std::get<outMemoryFormatIndex>(param);
                                                    return !containsSupportedFormatsOnly(inFormat, supported_f) ||
                                                           !containsSupportedFormatsOnly(outFormat, supported_f);
                                                 }),
                                filteredParamsVector.end());
    return filteredParamsVector;
}
} // namespace CPUTestUtils
