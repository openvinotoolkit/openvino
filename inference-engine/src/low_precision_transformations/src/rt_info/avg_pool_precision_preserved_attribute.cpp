// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/rt_info/avg_pool_precision_preserved_attribute.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include "low_precision/network_helper.hpp"

using namespace ngraph;

template class ngraph::VariantImpl<AvgPoolPrecisionPreservedAttributePtr>;

constexpr VariantTypeInfo VariantWrapper<AvgPoolPrecisionPreservedAttributePtr>::type_info;

std::shared_ptr<ngraph::Variant> VariantWrapper<AvgPoolPrecisionPreservedAttributePtr>::merge(const ngraph::NodeVector& nodes) {
    std::shared_ptr<::ngraph::VariantWrapper<AvgPoolPrecisionPreservedAttributePtr>> resultAttributeWrapper;

    for (const std::shared_ptr<ngraph::Node>& node : nodes) {
        auto attribute = ngraph::pass::low_precision::getAttribute<AvgPoolPrecisionPreservedAttributePtr>(node);
        if (attribute == nullptr) {
            continue;
        }

        if (resultAttributeWrapper == nullptr) {
            resultAttributeWrapper = attribute;
        }

        if (!attribute->get()->sharedValue->value) {
            return attribute;
        }
    }

    return resultAttributeWrapper;
}

void VariantWrapper<AvgPoolPrecisionPreservedAttributePtr>::merge(
    std::vector<std::shared_ptr<ngraph::VariantWrapper<std::shared_ptr<AvgPoolPrecisionPreservedAttribute>>>>& attributes) {
}

std::string VariantWrapper<AvgPoolPrecisionPreservedAttributePtr>::get_string() {
    auto value = this->m_value;
    std::stringstream ss;
    ss << m_value->get_string();
    ss << "value: " << (value->sharedValue->value ? "true" : "false");
    return ss.str();
}
