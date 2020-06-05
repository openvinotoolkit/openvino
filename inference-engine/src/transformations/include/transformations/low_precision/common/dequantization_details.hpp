// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <ngraph/check.hpp>
#include <transformations/low_precision/common/ie_lpt_exception.hpp>

namespace ngraph {
namespace pass {
namespace low_precision {

class DequantizationDetails {
public:
    DequantizationDetails(
        const std::vector<float>& scales,
        const std::vector<float>& shifts,
        const size_t channelsCount) :
        scales(scales), shifts(shifts), channelsCount(checkChannelsCount(channelsCount)) {}

    DequantizationDetails(
        const std::vector<float>& scales,
        const std::vector<float>& shifts) :
        scales(scales), shifts(shifts), channelsCount(checkChannelsCount(shifts.size())) {}

    size_t checkChannelsCount(const size_t channelsCount) {
        if ((scales.size() != shifts.size()) || (shifts.size() != channelsCount)) {
            THROW_TRANSFORMATION_EXCEPTION << "channels count is not correct \n";
        }
        return channelsCount;
    }

    const std::vector<float> scales;
    const std::vector<float> shifts;
    const size_t channelsCount;
};

}// namespace low_precision
}// namespace pass
}// namespace ngraph
