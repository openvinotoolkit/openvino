// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/rt_info/original_precision_attribute.hpp"

using namespace ov;

void ov::set_original_precision_attribute(const std::shared_ptr<Node>& node, const element::Type_t original_precision) {
    auto& rt_info = node->get_rt_info();
    rt_info[OriginalPrecisionAttribute::get_type_info_static()] = original_precision;
}

void ov::reset_original_precision_attribute(const std::shared_ptr<Node>& node) {
    auto& rt_info = node->get_rt_info();
    auto it = rt_info.find(OriginalPrecisionAttribute::get_type_info_static());
    if (it != rt_info.end()) {
        rt_info.erase(it);
    }
}

element::Type_t ov::get_original_precision(const std::shared_ptr<Node>& node) {
    auto& rt_info = node->get_rt_info();
    auto it = rt_info.find(OriginalPrecisionAttribute::get_type_info_static());
    if (it != rt_info.end()) {
        return it->second.as<element::Type_t>();
    } else {
        return element::Type_t::dynamic;
    }
}
