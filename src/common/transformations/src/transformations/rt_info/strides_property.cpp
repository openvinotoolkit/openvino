// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/rt_info/strides_property.hpp"

bool ov::has_strides_prop(const ngraph::Input<ngraph::Node>& node) {
    return node.get_rt_info().count(StridesPropagation::get_type_info_static());
}

ngraph::Strides ov::get_strides_prop(const ngraph::Input<ngraph::Node>& node) {
    return node.get_rt_info().at(StridesPropagation::get_type_info_static()).as<StridesPropagation>().value;
}

void ov::insert_strides_prop(ngraph::Input<ngraph::Node>& node, const ngraph::Strides& strides) {
    node.get_rt_info().emplace(StridesPropagation::get_type_info_static(), StridesPropagation{strides});
}
