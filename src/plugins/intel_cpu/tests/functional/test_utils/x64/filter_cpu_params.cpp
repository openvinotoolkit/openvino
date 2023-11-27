// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/filter_cpu_params.hpp"
#include "ie_ngraph_utils.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/rt_info/memory_formats_attribute.hpp"
#include "utils/general_utils.h"
#include <cstdint>

namespace CPUTestUtils {

std::vector<CPUSpecificParams> filterCPUSpecificParams(const std::vector<CPUSpecificParams> &paramsVector) {
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

    if (!InferenceEngine::with_cpu_x86_avx512f()) {
        for (auto& param : filteredParamsVector) {
            adjustBlockedFormatByIsa(std::get<0>(param));
            adjustBlockedFormatByIsa(std::get<1>(param));
        }
    }

    return filteredParamsVector;
}

} // namespace CPUTestUtils