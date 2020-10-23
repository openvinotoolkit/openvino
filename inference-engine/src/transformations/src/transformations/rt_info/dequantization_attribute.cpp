// Copyright (C) 2020 Intel Corporation
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

namespace ngraph {

template class ngraph::VariantImpl<DequantizationAttr>;

constexpr VariantTypeInfo VariantWrapper<DequantizationAttr>::type_info;

std::string DequantizationAttr::getDequantizationAttr() const {
    return dequantization_attribute;
}

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

std::string getDequantization(const std::shared_ptr<ngraph::Node>& node) {
    const auto& rtInfo = node->get_rt_info();
    using getDequantizationWraper = VariantWrapper<DequantizationAttr>;

    if (!rtInfo.count(getDequantizationWraper::type_info.name)) return "";

    const auto& attr = rtInfo.at(getDequantizationWraper::type_info.name);
    DequantizationAttr pp = as_type_ptr<getDequantizationWraper>(attr)->get();
    return pp.getDequantizationAttr();
}


}  // namespace ngraph
