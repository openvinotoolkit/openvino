// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/rt_info/precision_preserved_attribute.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include "low_precision/network_helper.hpp"

using namespace ngraph;

PrecisionPreservedAttribute::PrecisionPreservedAttribute(const bool value) {
    sharedValue->value = value;
}

//PrecisionPreservedAttribute::PrecisionPreservedAttribute(std::shared_ptr<SharedValue> value) : sharedValue(value) {
//    //
//}

template class ngraph::VariantImpl<PrecisionPreservedAttributePtr>;

constexpr VariantTypeInfo VariantWrapper<PrecisionPreservedAttributePtr>::type_info;

std::string VariantWrapper<PrecisionPreservedAttributePtr>::get_string() {
    auto& value = this->m_value;
    std::stringstream ss;
    ss << m_value->get_string();
    ss << "value: " << (value->sharedValue->value ? "true" : "false");
    return ss.str();
}
