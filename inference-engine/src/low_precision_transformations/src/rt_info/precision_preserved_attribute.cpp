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

template class ngraph::VariantImpl<PrecisionPreservedAttribute>;

constexpr VariantTypeInfo VariantWrapper<PrecisionPreservedAttribute>::type_info;

std::string VariantWrapper<PrecisionPreservedAttribute>::get_string() {
    auto& value = this->m_value;
    std::stringstream ss;

#ifdef _DEBUG
    const size_t rawPointer = (size_t)&this->m_value;
    ss << rawPointer << ": ";

    const size_t sharedRawPointer = (size_t)value.sharedValue.get();
    ss << "shared: " << sharedRawPointer << ",";
#endif

    ss << "value: " << (value.sharedValue->value ? "true" : "false");
    return ss.str();
}
