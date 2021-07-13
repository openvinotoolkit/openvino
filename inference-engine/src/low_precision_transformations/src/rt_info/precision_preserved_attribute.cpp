// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/rt_info/precision_preserved_attribute.hpp"

#include <memory>
#include <string>

using namespace ngraph;

PrecisionPreservedAttribute::PrecisionPreservedAttribute(const bool value) {
    sharedValue->value = value;
}

template class ngraph::VariantImpl<PrecisionPreservedAttributePtr>;

constexpr VariantTypeInfo VariantWrapper<PrecisionPreservedAttributePtr>::type_info;

std::string VariantWrapper<PrecisionPreservedAttributePtr>::to_string() {
    auto& value = this->m_value;
    std::stringstream ss;
    ss << m_value->get_string();
    ss << "value: " << (value->sharedValue->value ? "true" : "false");
    return ss.str();
}
