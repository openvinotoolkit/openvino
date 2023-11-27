// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/filter_cpu_params.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/rt_info/memory_formats_attribute.hpp"
#include "utils/general_utils.h"
#include <algorithm>

namespace CPUTestUtils {

std::vector<CPUSpecificParams> filterCPUSpecificParams(const std::vector<CPUSpecificParams> &paramsVector) {
    auto filterBlockedFormat = [](std::vector<cpu_memory_format_t>& formats) {
        formats.erase(std::remove_if(formats.begin(),
                      formats.end(),
                      [](cpu_memory_format_t f) {return !ov::intel_cpu::one_of(f,
                                                                               CPUTestUtils::cpu_memory_format_t::nwc,
                                                                               CPUTestUtils::cpu_memory_format_t::ncw,
                                                                               CPUTestUtils::cpu_memory_format_t::nchw,
                                                                               CPUTestUtils::cpu_memory_format_t::nhwc,
                                                                               CPUTestUtils::cpu_memory_format_t::ndhwc,
                                                                               CPUTestUtils::cpu_memory_format_t::ncdhw);
                                                 }),
                      formats.end());
    };

    std::vector<CPUSpecificParams> filteredParamsVector = paramsVector;

    for (auto& param : filteredParamsVector) {
        filterBlockedFormat(std::get<0>(param));
        filterBlockedFormat(std::get<1>(param));
    }

    return filteredParamsVector;
}

} // namespace CPUTestUtils