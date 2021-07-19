// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/rt_info/avg_pool_precision_preserved_attribute.hpp"

#include <memory>
#include <vector>
#include <ngraph/variant.hpp>

using namespace ngraph;

template class ngraph::VariantImpl<AvgPoolPrecisionPreservedAttributePtr>;

constexpr VariantTypeInfo VariantWrapper<AvgPoolPrecisionPreservedAttributePtr>::type_info;

void VariantWrapper<AvgPoolPrecisionPreservedAttributePtr>::merge(
    std::vector<std::shared_ptr<ngraph::VariantWrapper<std::shared_ptr<AvgPoolPrecisionPreservedAttribute>>>>& attributes) {
}

std::string VariantWrapper<AvgPoolPrecisionPreservedAttributePtr>::to_string() {
    auto value = this->m_value;
    std::stringstream ss;
    ss << m_value->get_string();
    ss << "value: " << (value->sharedValue->value ? "true" : "false");
    return ss.str();
}
