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

        if (selectedTypeStr.find("acl") != std::string::npos)
            continue;

        resCPUParams.push_back(param);
    }

    return resCPUParams;
}

std::vector<CPUSpecificParams> filterCPUInfoForDevice(const std::vector<CPUSpecificParams>& CPUParams) {
    std::vector<CPUSpecificParams> resCPUParams;
    const int selectedTypeIndex = 3;
    const int inputFormatIndex = 0;

    for (auto param : CPUParams) {
        auto selectedTypeStr = std::get<selectedTypeIndex>(param);
        auto inputsFormat = std::get<inputFormatIndex>(param);
        if (!inputsFormat.empty() && !selectedTypeStr.empty() && selectedTypeStr == "any_type") {
            if (ov::intel_cpu::one_of(inputsFormat[0], nCw8c, nChw8c, nCdhw8c) && !ov::with_cpu_x86_sse42())
                continue;
            if (ov::intel_cpu::one_of(inputsFormat[0], nCw16c, nChw16c, nCdhw16c) && !ov::with_cpu_x86_avx512f())
                continue;
        }
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

std::vector<CPUSpecificParams> filterCPUInfoForDeviceWithFP16(const std::vector<CPUSpecificParams>& allParams) {
    std::vector<CPUSpecificParams> specificParams;
    if (!ov::with_cpu_x86_avx512_core_fp16()) {
        return specificParams;
    }
    std::copy_if(allParams.begin(), allParams.end(), std::back_inserter(specificParams), [](const CPUSpecificParams& item) {
        const auto &selected = std::get<3>(item);
        bool isValid = false;
        if (selected.find("avx512") != std::string::npos) {
            isValid = true;
        }
        if ((!ov::with_cpu_x86_avx512_core_amx_fp16()) && selected.find("amx") != std::string::npos) {
            isValid = false;
        }
        return isValid;
    });
    return specificParams;
}

std::vector<CPUSpecificParams> filterCPUSpecificParams(const std::vector<CPUSpecificParams>& paramsVector) {
    auto adjustBlockedFormatByIsa = [](std::vector<cpu_memory_format_t>& formats) {
        for (auto& format : formats) {
            if (format == nCw16c)
                format = nCw8c;
            if (format == nChw16c)
                format = nChw8c;
            if (format == nCdhw16c)
                format = nCdhw8c;
        }
    };

    std::vector<CPUSpecificParams> filteredParamsVector = paramsVector;

    if (!ov::with_cpu_x86_avx512f()) {
        for (auto& param : filteredParamsVector) {
            adjustBlockedFormatByIsa(std::get<0>(param));
            adjustBlockedFormatByIsa(std::get<1>(param));
        }
    }

    return filteredParamsVector;
}
} // namespace CPUTestUtils
