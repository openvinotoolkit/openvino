// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/rt_info/strides_property.hpp"

bool ov::has_strides_prop(const ov::Input<ov::Node>& node) {
    return node.get_rt_info().count(StridesPropagation::get_type_info_static());
}

ov::Strides ov::get_strides_prop(const ov::Input<ov::Node>& node) {
    return node.get_rt_info().at(StridesPropagation::get_type_info_static()).as<StridesPropagation>().value;
}

void ov::insert_strides_prop(ov::Input<ov::Node>& node, const ov::Strides& strides) {
    node.get_rt_info().emplace(StridesPropagation::get_type_info_static(), StridesPropagation{strides});
}

void ov::remove_strides_prop(ov::Input<ov::Node>& node) {
    auto& rt_info = node.get_rt_info();
    auto it = rt_info.find(StridesPropagation::get_type_info_static());
    if (it != rt_info.end()) {
        rt_info.erase(it);
    }
}
