// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/rt_info/precisions_attribute.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <iterator>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include "low_precision/network_helper.hpp"

using namespace ngraph;

PrecisionsAttribute::PrecisionsAttribute(const std::set<ngraph::element::Type>& precisions) {
    sharedValue->precisions = precisions;
}

template class ngraph::VariantImpl<std::shared_ptr<PrecisionsAttribute>>;

constexpr VariantTypeInfo VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info;

std::shared_ptr<ngraph::Variant> VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::merge(const ngraph::NodeVector& nodes) {
    return nullptr;
}

void VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::merge(
    std::vector<std::shared_ptr<VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>>& attributes) {
    auto my = this->get()->sharedValue->precisions;

    for (auto attribute : attributes) {
        auto attributeValues = attribute->get()->sharedValue->precisions;
        std::set<element::Type> result;
        set_intersection(
            attributeValues.begin(),
            attributeValues.end(),
            my.begin(),
            my.end(),
            std::inserter(result, result.begin()));
        my = result;
    }

    this->get()->sharedValue->precisions = my;
}

std::shared_ptr<ngraph::Variant> VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::init(const std::shared_ptr<ngraph::Node>& node) {
    return nullptr;
}

std::string VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::get_string() {
    std::stringstream ss;

#ifdef _DEBUG
    const size_t rawPointer = (size_t)m_value.get();
    ss << rawPointer << ": ";

    const size_t sharedValueRawPointer = (size_t)m_value->sharedValue.get();
    ss << "sharedValue: " << sharedValueRawPointer;

    bool firstAttribute = true;
    ss << ", attributes: [";
    for (auto& attributeWeakPtr : m_value->sharedValue->attributes) {
        auto attribute = attributeWeakPtr.lock();
        if (attribute == nullptr) {
            continue;
        }

        if (!firstAttribute) {
            ss << ", ";
        }
        ss << (size_t)attribute.get();
        firstAttribute = false;
    }
    ss << "], ";
#endif

    bool firstPrecision = true;

    ss << "precisions: [";
    for (const auto& value : m_value->sharedValue->precisions) {
        if (!firstPrecision) {
            ss << ", ";
        }
        ss << value;
        firstPrecision = false;
    }
    ss << "]";

    return ss.str();
}
