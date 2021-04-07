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

template class ngraph::VariantImpl<std::shared_ptr<PrecisionsAttribute>>;

constexpr VariantTypeInfo VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info;

std::shared_ptr<ngraph::Variant> VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::merge(const ngraph::NodeVector& nodes) {
    return nullptr;
}

void VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::merge(std::vector<std::shared_ptr<VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>>& attributes) {
    auto my = this->get()->precisions;

    for (auto attribute : attributes) {
        auto attributeValues = attribute->get()->precisions;
        std::set<element::Type> result;
        set_intersection(
            attributeValues.begin(),
            attributeValues.end(),
            my.begin(),
            my.end(),
            std::inserter(result, result.begin()));
        my = result;
    }

    this->get()->precisions = my;
}

std::shared_ptr<ngraph::Variant> VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::init(const std::shared_ptr<ngraph::Node>& node) {
    return nullptr;
}

std::string VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::get_string() {
    std::stringstream ss;

#ifdef _DEBUG
    const size_t rawPointer = (size_t)m_value.get();
    ss << rawPointer << ": ";
#endif

    bool first = true;
    for (const auto& value : m_value->precisions) {
        if (!first) {
            ss << ", ";
        }
        ss << value;
        first = false;
    }
    return ss.str();
}
