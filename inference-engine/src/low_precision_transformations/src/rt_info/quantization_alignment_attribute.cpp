// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/rt_info/quantization_alignment_attribute.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include "low_precision/network_helper.hpp"

using namespace ngraph;
using namespace ngraph::pass::low_precision;

QuantizationAlignmentAttribute::QuantizationAlignmentAttribute(const bool hasToBeAligned) {
    sharedValue = std::make_shared<QuantizationAlignmentSharedValue>(hasToBeAligned);
}

template class ngraph::VariantImpl<QuantizationAlignmentAttributePtr>;

constexpr VariantTypeInfo VariantWrapper<QuantizationAlignmentAttributePtr>::type_info;

std::shared_ptr<ngraph::Variant> VariantWrapper<QuantizationAlignmentAttributePtr>::init(const std::shared_ptr<ngraph::Node>& node) {
    return nullptr;
}

std::shared_ptr<VariantWrapper<std::shared_ptr<QuantizationAlignmentAttribute>>> VariantWrapper<QuantizationAlignmentAttributePtr>::create(
    const std::shared_ptr<ngraph::Node>& node,
    const AttributeParameters& params) {
    if (getAttribute<std::shared_ptr<QuantizationAlignmentAttribute>>(node) != nullptr) {
        return nullptr;
    }

    if (!NetworkHelper::isPrecisionPreserved(node)) {
        return nullptr;
    }

    bool leastOneOperationIsFakeQuantize = false;
    bool leastOneOperationIsNotFakeQuantize = false;
    for (auto index = 0ul; index < node->get_input_size(); ++index) {
        const auto& input = node->input(index);
        auto inputNode = input.get_source_output().get_node_shared_ptr();

        const auto dequantization = NetworkHelper::getDequantization(node, index);
        if (!dequantization.empty() &&
            (is_type<opset1::Convert>(dequantization.data.get_node())) &&
            is_type<opset1::FakeQuantize>(dequantization.data.get_node()->get_input_node_ptr(0))) {
            inputNode = dequantization.data.get_node()->get_input_node_shared_ptr(0);
        }

        if (is_type<opset1::Constant>(inputNode)) {
            continue;
        }

        if (!is_type<opset1::FakeQuantize>(inputNode)) {
            leastOneOperationIsNotFakeQuantize = true;
            break;
        }

        leastOneOperationIsFakeQuantize = true;
    }

    if (leastOneOperationIsFakeQuantize && !leastOneOperationIsNotFakeQuantize) {
        auto& rt = node->get_rt_info();
        const auto attribute = std::make_shared<ngraph::VariantWrapper<QuantizationAlignmentAttributePtr>>(
            make_shared_attribute<QuantizationAlignmentAttribute>());
        rt[ngraph::VariantWrapper<QuantizationAlignmentAttributePtr>::type_info.name] = attribute;
        return attribute;
    }

    return nullptr;
}

void VariantWrapper<QuantizationAlignmentAttributePtr>::merge(
    std::vector<std::shared_ptr<VariantWrapper<std::shared_ptr<QuantizationAlignmentAttribute>>>>& attributes) {
    auto currentAttributte = get();
    for (const auto& attribute : attributes) {
        currentAttributte->sharedValue->value = currentAttributte->sharedValue->value || attribute->get()->sharedValue->value;
    }
}

std::string VariantWrapper<QuantizationAlignmentAttributePtr>::to_string() {
    std::stringstream ss;
    ss << m_value->get_string();
    ss << "value: " << (m_value->sharedValue->value ? "true" : "false");
    return ss.str();
}
