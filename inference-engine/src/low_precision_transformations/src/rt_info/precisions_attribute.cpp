// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/rt_info/precisions_attribute.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include "low_precision/network_helper.hpp"

using namespace ngraph;

template class ngraph::VariantImpl<PrecisionsAttribute>;

constexpr VariantTypeInfo VariantWrapper<PrecisionsAttribute>::type_info;

std::shared_ptr<ngraph::Variant> VariantWrapper<PrecisionsAttribute>::merge(const ngraph::NodeVector& nodes) {
    return nullptr;
}

std::shared_ptr<ngraph::Variant> VariantWrapper<PrecisionsAttribute>::init(const std::shared_ptr<ngraph::Node>& node) {
    return nullptr;
}

std::string VariantWrapper<PrecisionsAttribute>::get_string() {
    std::stringstream ss;
    ss << "{";
    bool first = true;
    for (const auto& value : m_value.sharedPart->value->precisions) {
        if (!first) {
            ss << ", ";
        }
        ss << value;
        first = true;
    }
    ss << "}";
    return ss.str();
}
