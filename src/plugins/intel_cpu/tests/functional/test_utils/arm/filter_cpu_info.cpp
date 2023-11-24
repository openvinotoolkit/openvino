// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/filter_cpu_info.hpp"
#include "ie_ngraph_utils.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/rt_info/memory_formats_attribute.hpp"
#include "utils/general_utils.h"
#include <cstdint>

namespace CPUTestUtils {

std::vector<CPUSpecificParams> filterCPUInfo(const std::vector<CPUSpecificParams>& CPUParams) {
    std::vector<CPUSpecificParams> archCPUParams = filterCPUInfoForArch(CPUParams);
    std::vector<CPUSpecificParams> deviceCPUParams = filterCPUInfoForDevice(archCPUParams);
    return deviceCPUParams;
}

std::vector<CPUSpecificParams> filterCPUInfoForArch(const std::vector<CPUSpecificParams>& CPUParams) {
    std::vector<CPUSpecificParams> aclResCPUParams;
    std::vector<CPUSpecificParams> refResCPUParams;
    const int selectedTypeIndex = 3;

    for (auto param : CPUParams) {
        auto selectedTypeStr = std::get<selectedTypeIndex>(param);

        if (selectedTypeStr.find("acl") != std::string::npos) {
            aclResCPUParams.push_back(param);
            continue;
        }
        if (selectedTypeStr.find("ref") != std::string::npos) {
            refResCPUParams.push_back(param);
            continue;
        }
    }
    // if acl type is found then ref types are not going to be tested
    return aclResCPUParams.empty() ? refResCPUParams : aclResCPUParams;
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

} // namespace CPUTestUtils
