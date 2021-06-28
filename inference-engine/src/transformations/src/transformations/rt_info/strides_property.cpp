// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/rt_info/strides_property.hpp"

constexpr ngraph::VariantTypeInfo ngraph::VariantWrapper<ngraph::Strides>::type_info;

bool has_strides_prop(const ngraph::Input<ngraph::Node>& node) {
    const auto& rt_map = node.get_rt_info();
    auto it = rt_map.find(ngraph::VariantWrapper<ngraph::Strides>::type_info.name);
    return it != rt_map.end();
}

ngraph::Strides get_strides_prop(const ngraph::Input<ngraph::Node>& node) {
    const auto& rt_map = node.get_rt_info();
    const auto& var = rt_map.at(ngraph::VariantWrapper<ngraph::Strides>::type_info.name);
    return ngraph::as_type_ptr<ngraph::VariantWrapper<ngraph::Strides>>(var)->get();
}

void insert_strides_prop(ngraph::Input<ngraph::Node>& node, const ngraph::Strides& strides) {
    auto& rt_map = node.get_rt_info();
    rt_map[ngraph::VariantWrapper<ngraph::Strides>::type_info.name] = std::make_shared<ngraph::VariantWrapper<ngraph::Strides>>(strides);
}
