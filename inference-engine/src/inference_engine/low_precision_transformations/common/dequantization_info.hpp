// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <cstddef>

namespace InferenceEngine {
namespace details {

class DequantizationInfo {
public:
    DequantizationInfo(
        const size_t levels,
        const std::vector<float>& outputLowValues,
        const std::vector<float>& outputHighValues);

    size_t outputChannels() const;

    const size_t levels;
    const std::vector<float> outputLowValues;
    const std::vector<float> outputHighValues;
};

}  // namespace details
}  // namespace InferenceEngine
