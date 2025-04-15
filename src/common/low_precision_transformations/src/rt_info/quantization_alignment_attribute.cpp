// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/rt_info/quantization_alignment_attribute.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "openvino/opsets/opset1.hpp"
#include "low_precision/network_helper.hpp"

using namespace ov;
using namespace ov;
using namespace ov::pass::low_precision;

QuantizationAlignmentAttribute::QuantizationAlignmentAttribute(const bool hasToBeAligned) :
    SharedAttribute(hasToBeAligned) {
}

ov::Any QuantizationAlignmentAttribute::create(
    const std::shared_ptr<ov::Node>& node,
    const AttributeParameters& params) {
    if (!getAttribute<QuantizationAlignmentAttribute>(node).empty()) {
        return {};
    }

    if (!NetworkHelper::isPrecisionPreserved(node)) {
        return {};
    }

    bool leastOneOperationIsFakeQuantize = false;
    bool leastOneOperationIsNotFakeQuantize = false;
    for (auto index = 0ul; index < node->get_input_size(); ++index) {
        const auto& input = node->input(index);
        auto inputNode = input.get_source_output().get_node_shared_ptr();

        const auto dequantization = NetworkHelper::getDequantization(node, params.defaultPrecisions, index);
        if (!dequantization.empty() &&
            (ov::is_type<opset1::Convert>(dequantization.data.get_node())) &&
            ov::is_type<opset1::FakeQuantize>(dequantization.data.get_node()->get_input_node_ptr(0))) {
            inputNode = dequantization.data.get_node()->get_input_node_shared_ptr(0);
        }

        if (ov::is_type<opset1::Constant>(inputNode)) {
            continue;
        }

        if (!ov::is_type<opset1::FakeQuantize>(inputNode)) {
            leastOneOperationIsNotFakeQuantize = true;
            break;
        }

        leastOneOperationIsFakeQuantize = true;
    }

    if (leastOneOperationIsFakeQuantize && !leastOneOperationIsNotFakeQuantize) {
        auto& rt = node->get_rt_info();
        rt[QuantizationAlignmentAttribute::get_type_info_static()] = QuantizationAlignmentAttribute();
        return rt[QuantizationAlignmentAttribute::get_type_info_static()];
    }

    return {};
}

void QuantizationAlignmentAttribute::merge_attributes(std::vector<ov::Any>& attributes) {
    for (const auto& other_attribute : attributes) {
        value() = value() || other_attribute.as<QuantizationAlignmentAttribute>().value();
    }
}

std::string QuantizationAlignmentAttribute::to_string() const {
    std::stringstream ss;
    ss << attribute->get_string();
    ss << "value: " << (value() ? "true" : "false");
    return ss.str();
}
