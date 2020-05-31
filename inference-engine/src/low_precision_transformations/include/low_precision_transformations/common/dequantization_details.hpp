// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <details/ie_exception.hpp>

namespace InferenceEngine {
namespace details {

IE_SUPPRESS_DEPRECATED_START

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
            THROW_IE_EXCEPTION << "channels count is not correct";
        }
        return channelsCount;
    }

    bool isPerTensor() const {
        if ((scales.size() == 0) || (shifts.size() == 0)) {
            THROW_IE_EXCEPTION << "scale or shift values count is not correct";
        }
        return
            std::all_of(scales.begin(), scales.end(), [&](const float value) { return value == scales[0]; }) &&
            std::all_of(shifts.begin(), shifts.end(), [&](const float value) { return value == shifts[0]; });
    }

    const std::vector<float> scales;
    const std::vector<float> shifts;
    const size_t channelsCount;
};

IE_SUPPRESS_DEPRECATED_END

}  // namespace details
}  // namespace InferenceEngine
