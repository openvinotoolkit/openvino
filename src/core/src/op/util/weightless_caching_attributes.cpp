// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/rt_info/weightless_caching_attributes.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"

bool ov::WeightlessCacheAttribute::is_copyable() const {
    return false;
}

OPENVINO_API void ov::copy_weightless_cache_attr(const std::shared_ptr<ov::Node>& from,
                                                 const std::shared_ptr<ov::Node>& to) {
    const auto& rt_info = from->get_rt_info();
    auto weightless_caching_attr = rt_info.find(ov::WeightlessCacheAttribute::get_type_info_static());

    if (weightless_caching_attr != rt_info.end()) {
        to->get_rt_info()[ov::WeightlessCacheAttribute::get_type_info_static()] = weightless_caching_attr->second;
    } else if (ov::is_type<ov::op::v0::Convert>(from) && ov::is_type<ov::op::v0::Constant>(to)) {
        auto convert_node = ov::as_type_ptr<ov::op::v0::Convert>(from);
        auto const_node = convert_node->get_input_node_ptr(0);
        weightless_caching_attr = const_node->get_rt_info().find(ov::WeightlessCacheAttribute::get_type_info_static());
        if (weightless_caching_attr != const_node->get_rt_info().end()) {
            to->get_rt_info()[ov::WeightlessCacheAttribute::get_type_info_static()] = weightless_caching_attr->second;
        }
    }
}