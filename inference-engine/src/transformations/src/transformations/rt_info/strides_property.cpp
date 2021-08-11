// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/rt_info/strides_property.hpp"

constexpr ov::VariantTypeInfo ov::VariantWrapper<ov::Strides>::type_info;

bool has_strides_prop(const ov::Input<ov::Node>& node) {
    const auto& rt_map = node.get_rt_info();
    auto it = rt_map.find(ov::VariantWrapper<ov::Strides>::type_info.name);
    return it != rt_map.end();
}

ov::Strides get_strides_prop(const ov::Input<ov::Node>& node) {
    const auto& rt_map = node.get_rt_info();
    const auto& var = rt_map.at(ov::VariantWrapper<ov::Strides>::type_info.name);
    return ov::as_type_ptr<ov::VariantWrapper<ov::Strides>>(var)->get();
}

void insert_strides_prop(ov::Input<ov::Node>& node, const ov::Strides& strides) {
    auto& rt_map = node.get_rt_info();
    rt_map[ov::VariantWrapper<ov::Strides>::type_info.name] = std::make_shared<ov::VariantWrapper<ov::Strides>>(strides);
}
