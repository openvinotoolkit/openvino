// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/rt_info/keep_const_precision.hpp"

void ov::enable_keep_const_precision(const std::shared_ptr<Node>& node) {
    auto& rt_info = node->get_rt_info();
    rt_info[KeepConstPrecision::get_type_info_static()] = KeepConstPrecision{};
}

void ov::disable_keep_const_precision(const std::shared_ptr<Node>& node) {
    auto& rt_info = node->get_rt_info();
    rt_info.erase(KeepConstPrecision::get_type_info_static());
}

bool ov::is_keep_const_precision(const std::shared_ptr<const Node>& node) {
    const auto& rt_info = node->get_rt_info();
    return rt_info.count(KeepConstPrecision::get_type_info_static());
}
