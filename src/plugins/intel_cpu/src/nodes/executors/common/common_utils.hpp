// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// @file common_utils.hpp
// Contains utility methods used by all executors
//

#pragma once

#include <vector>

#include "nodes/executors/memory_arguments.hpp"
#include "utils/cpp/maybe_unused.hpp"
#include "utils/cpu_utils.hpp"

namespace ov::intel_cpu {

OV_CPU_MAYBE_UNUSED_FUNCTION static std::vector<float> getDeQuantizedScales(const MemoryArgs& memory) {
    if (!memory.count(ARG_DST_DEQ_SCALE)) {
        return {};
    }

    auto scalesMemory = memory.at(ARG_DST_DEQ_SCALE);

    auto scalesData = static_cast<const float*>(scalesMemory->getData());

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
