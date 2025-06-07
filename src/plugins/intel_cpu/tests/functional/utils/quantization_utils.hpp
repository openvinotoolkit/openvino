// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <unordered_map>
#include "openvino/op/util/attr_types.hpp"

namespace CPUTestUtils {

struct QuantizationData {
    float il;
    float ih;
    float ol;
    float oh;
    int levels = -1;  // -1 means that levels are not defined
    bool isPerChannel = false;
    ov::op::AutoBroadcastSpec auto_broadcast = ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::NUMPY);
};

struct QuantizationInfo {
    std::unordered_map<size_t, QuantizationData> inputs;
    std::unordered_map<size_t, QuantizationData> outputs;

    bool empty() const {
        return inputs.empty() && outputs.empty();
    }
};

inline std::ostream& operator<<(std::ostream& os, const QuantizationData& qdata) {
    os << qdata.il << "_" << qdata.ih << "_" << qdata.ol << "_" << qdata.oh << "_levels=" << qdata.levels
       << "_perChannel=" << std::boolalpha << qdata.isPerChannel;
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
