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

// order defines default precision
const std::vector<ngraph::element::Type> PrecisionsAttribute::defaultPrecisions = { ngraph::element::u8, ngraph::element::i8 };

PrecisionsAttribute::PrecisionsAttribute(const std::vector<ngraph::element::Type>& precisions) {
    sharedValue->precisions = precisions;
}

template class ngraph::VariantImpl<std::shared_ptr<PrecisionsAttribute>>;

constexpr VariantTypeInfo VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info;

std::shared_ptr<VariantWrapper<std::shared_ptr<PrecisionsAttribute>>> VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::create(
    const std::shared_ptr<ngraph::Node>& node,
    const AttributeParameters& params) {
    auto attribute = ngraph::pass::low_precision::make_shared_attribute<PrecisionsAttribute>();
    auto wrapper = std::make_shared<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>(attribute);

    auto& rt = is_type<opset1::FakeQuantize>(node) ? node->output(0).get_rt_info() : node->get_rt_info();
    rt[ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info.name] = wrapper;
    return wrapper;
}

void VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::merge(
    std::vector<std::shared_ptr<VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>>& attributes) {
    auto& my = this->get()->sharedValue->precisions;
    for (auto attribute : attributes) {
        const auto& attributeValues = attribute->get()->sharedValue->precisions;
        auto it = my.begin();
        while (it != my.end()) {
            if (std::find(attributeValues.begin(), attributeValues.end(), *it) == attributeValues.end()) {
                it = my.erase(it);
            } else {
                it++;
            }
        }
        if (my.size() == 0ul) {
            break;
        }
    }
}

std::shared_ptr<ngraph::Variant> VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::init(const std::shared_ptr<ngraph::Node>& node) {
    return nullptr;
}

std::string VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::to_string() {
    std::stringstream ss;

    ss << m_value->get_string();

    bool firstPrecision = true;
    ss << "precisions: {";
    for (const auto& value : m_value->sharedValue->precisions) {
        if (!firstPrecision) {
            ss << ", ";
        }
        ss << value;
        firstPrecision = false;
    }
    ss << "}";

    return ss.str();
}
