// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/rt_info/strides_property.hpp"

bool ov::has_strides_prop(const ngraph::Input<ngraph::Node>& node) {
    const auto& rt_map = node.get_rt_info();
    return rt_map.count(StridesPropagation::get_type_info_static());
}

ngraph::Strides ov::get_strides_prop(const ngraph::Input<ngraph::Node>& node) {
    const auto& rt_map = node.get_rt_info();
    const auto& var = rt_map.at(StridesPropagation::get_type_info_static());
    return ngraph::as_type_ptr<StridesPropagation>(var)->get();
}

void ov::insert_strides_prop(ngraph::Input<ngraph::Node>& node, const ngraph::Strides& strides) {
    auto& rt_map = node.get_rt_info();
    rt_map[StridesPropagation::get_type_info_static()] = std::make_shared<StridesPropagation>(strides);
}
