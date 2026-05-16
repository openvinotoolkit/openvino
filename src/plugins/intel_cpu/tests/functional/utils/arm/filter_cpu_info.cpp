// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/system_conf.hpp"
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

        if (selectedTypeStr.find("acl") == std::string::npos &&
            selectedTypeStr.find("ref") == std::string::npos &&
            selectedTypeStr.find("kleidiai") == std::string::npos)
            continue;
#if defined(OPENVINO_ARCH_ARM)
        // disable gemm_acl on 32-bit arm platforms because oneDNN\ACL does not support it
        if (selectedTypeStr.find("gemm_acl") != std::string::npos)
            continue;
#endif
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
    std::vector<CPUSpecificParams> specificParams;
    if (!ov::with_cpu_neon_fp16()) {
        return specificParams;
    }
    std::copy_if(allParams.begin(), allParams.end(), std::back_inserter(specificParams), [](const CPUSpecificParams& item) {
        const auto& selected = std::get<3>(item);
        return selected.find("acl") != std::string::npos;
    });
    return specificParams;
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
