// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <assert.h>
#include <functional>
#include <memory>
#include <iterator>
#include <ostream>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>

#include "transformations/rt_info/dequantization_attribute.hpp"

using namespace ov;
using namespace ngraph;

std::string DequantizationAttr::getDequantizationAttr() const {
    return dequantization_attribute;
}

std::string ngraph::getDequantization(const std::shared_ptr<ngraph::Node>& node) {
    const auto& rtInfo = node->get_rt_info();
    using getDequantizationWrapper = VariantWrapper<DequantizationAttr>;

    if (!rtInfo.count(getDequantizationWrapper::type_info.name)) return "";

    const auto& attr = rtInfo.at(getDequantizationWrapper::type_info.name);
    DequantizationAttr pp = ov::as_type_ptr<getDequantizationWrapper>(attr)->get();
    return pp.getDequantizationAttr();
}

template class ov::VariantImpl<DequantizationAttr>;

constexpr VariantTypeInfo VariantWrapper<DequantizationAttr>::type_info;

std::shared_ptr<ngraph::Variant> VariantWrapper<DequantizationAttr>::merge(const ngraph::NodeVector & nodes) {
    std::set<std::string> dequantizations;

    for (auto& node : nodes) {
        std::string pp = getDequantization(node);
        if (!pp.empty()) dequantizations.insert(pp);
    }

    std::string final_primitives_priority;
    if (dequantizations.size() == 0) {
        final_primitives_priority = "";
    } else {
        final_primitives_priority = *dequantizations.begin();
    }
    return std::make_shared<VariantWrapper<DequantizationAttr>>(DequantizationAttr(final_primitives_priority));
}

std::shared_ptr<ngraph::Variant> VariantWrapper<DequantizationAttr>::init(const std::shared_ptr<ngraph::Node> & node) {
    return std::make_shared<VariantWrapper<DequantizationAttr>>(DequantizationAttr(node->get_friendly_name()));
}
