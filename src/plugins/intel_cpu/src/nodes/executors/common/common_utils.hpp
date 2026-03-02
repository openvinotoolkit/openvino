// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// @file common_utils.hpp
// Contains utility methods used by all executors
//

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>

#include "nodes/executors/memory_arguments.hpp"
#include "openvino/core/except.hpp"
#include "utils/cpu_utils.hpp"

namespace ov::intel_cpu {

inline bool isPerTensorDataWithTolerance(const std::vector<float>& values,
                                         const float tolerance = std::numeric_limits<float>::min()) {
    if (values.empty()) {
        return true;
    }

    const auto ref = values.front();
    return std::all_of(values.cbegin(), values.cend(), [&](const float value) {
        return std::fabs(value - ref) < tolerance;
    });
}

inline void multiplyAndBroadcastScales(std::vector<float>& dstScales,
                                       const std::vector<float>& srcScales,
                                       const std::size_t channelCount) {
    OPENVINO_ASSERT(any_of(srcScales.size(), static_cast<std::size_t>(1), channelCount),
                    "Invalid source scales size: ",
                    srcScales.size(),
                    ", expected 1 or ",
                    channelCount);

    if (dstScales.empty()) {
        dstScales.assign(srcScales.size() > 1 ? channelCount : static_cast<std::size_t>(1), 1.0F);
    }

    OPENVINO_ASSERT(any_of(dstScales.size(), static_cast<std::size_t>(1), channelCount),
                    "Invalid destination scales size: ",
                    dstScales.size(),
                    ", expected 1 or ",
                    channelCount);

    if (dstScales.size() == 1 && srcScales.size() > 1) {
        dstScales.assign(channelCount, dstScales.front());
    }

    if (srcScales.size() == 1) {
        for (auto& dstScale : dstScales) {
            dstScale *= srcScales.front();
        }
        return;
    }

    OPENVINO_ASSERT(dstScales.size() == channelCount,
                    "Invalid destination scales size after broadcasting: ",
                    dstScales.size(),
                    ", expected ",
                    channelCount);

    for (size_t i = 0; i < channelCount; i++) {
        dstScales[i] *= srcScales[i];
    }
}

[[maybe_unused]] static std::vector<float> getDeQuantizedScales(const MemoryArgs& memory) {
    if (memory.find(ARG_DST_DEQ_SCALE) == memory.end()) {
        return {};
    }

    auto scalesMemory = memory.at(ARG_DST_DEQ_SCALE);

    const auto* scalesData = static_cast<const float*>(scalesMemory->getData());

    if (!scalesData) {
        return {};
    }

    auto dstShape = memory.at(ARG_DST)->getShape();
    auto dqScalesShape = scalesMemory->getShape();

    auto scalesDims = getNormalizedDimsBySize(dqScalesShape.getDims(), dstShape.getDims().size());

    auto scaleSize =
        std::accumulate(scalesDims.begin(), scalesDims.end(), static_cast<std::size_t>(1), std::multiplies<>());

    std::vector<float> DQScales(scaleSize, 1.0);

    OPENVINO_ASSERT(scaleSize == 1 || DQScales.size() == 1 || DQScales.size() == scaleSize,
                    "set invalid scales size , DQScales vector size: ",
                    DQScales.size(),
                    ", scale data size: ",
                    scaleSize);

    // @todo do we really need to broadcast dq scales and then resize them back?
    if (scaleSize > DQScales.size()) {
        DQScales.resize(scaleSize, DQScales[0]);
    }
    if (1 == scaleSize) {
        std::transform(DQScales.begin(), DQScales.end(), DQScales.begin(), [=](float val) {
            return (scalesData[0] * val);
        });
    } else {
        for (size_t i = 0; i < DQScales.size(); i++) {
            DQScales[i] *= scalesData[i];
        }
    }
    if (std::all_of(DQScales.begin(), DQScales.end(), [&](float val) {
            return (val == DQScales[0]);
        })) {
        DQScales.resize(1);
    }

    return DQScales;
}

}  // namespace ov::intel_cpu
