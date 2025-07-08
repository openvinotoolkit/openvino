// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ostream>
#include <unordered_map>

namespace CPUTestUtils {

struct QuantizationData {
    QuantizationData(float il, float ih, float ol, float oh, int levels)
        : il(il),
          ih(ih),
          ol(ol),
          oh(oh),
          levels(levels) {}

    QuantizationData(float il, float ih, int levels) : il(il), ih(ih), ol(il), oh(ih), levels(levels) {}

    float il;
    float ih;
    float ol;
    float oh;
    int levels;
};

struct QuantizationInfo {
    std::unordered_map<size_t, QuantizationData> inputs;
    std::unordered_map<size_t, QuantizationData> outputs;

    bool empty() const {
        return inputs.empty() && outputs.empty();
    }
};

inline std::ostream& operator<<(std::ostream& os, const QuantizationData& qdata) {
    os << qdata.il << "_" << qdata.ih << "_" << qdata.ol << "_" << qdata.oh << "_levels=" << qdata.levels;
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const QuantizationInfo& qinfo) {
    os << "QuantizationInfo_[";
    os << "inputs_[";
    if (!qinfo.inputs.empty()) {
        for (const auto& [inputId, qData] : qinfo.inputs) {
            os << inputId << "_[" << qData << "]_";
        }
    }
    os << "]_";

    os << "outputs_[";
    if (!qinfo.outputs.empty()) {
        for (const auto& [outputId, qData] : qinfo.inputs) {
            os << outputId << "_[" << qData << "]_";
        }
    }
    os << "]";

    return os;
}

}  // namespace CPUTestUtils
