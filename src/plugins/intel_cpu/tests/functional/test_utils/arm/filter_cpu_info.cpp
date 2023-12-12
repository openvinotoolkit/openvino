// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/filter_cpu_info.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/rt_info/memory_formats_attribute.hpp"
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
            selectedTypeStr.find("ref") == std::string::npos)
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
    return CPUParams;
}

} // namespace CPUTestUtils
