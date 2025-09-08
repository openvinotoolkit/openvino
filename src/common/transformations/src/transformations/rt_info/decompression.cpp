// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/rt_info/decompression.hpp"

void ov::mark_as_decompression(const std::shared_ptr<Node>& node) {
    auto& rt_info = node->get_rt_info();
    rt_info[Decompression::get_type_info_static()] = Decompression();
}

void ov::unmark_as_decompression(const std::shared_ptr<Node>& node) {
    auto& rt_info = node->get_rt_info();
    rt_info.erase(Decompression::get_type_info_static());
}

bool ov::is_decompression(const std::shared_ptr<Node>& node) {
    const auto& rt_info = node->get_rt_info();
    return rt_info.count(Decompression::get_type_info_static());
}
