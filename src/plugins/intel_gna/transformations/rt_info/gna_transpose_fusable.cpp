// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gna_transpose_fusable.hpp"

void ov::intel_gna::rt_info::add_transpose_fusable(const std::shared_ptr<Node>& node) {
    auto& rt_info = node->get_rt_info();
    rt_info[GNATransposeFusable::get_type_info_static()] = std::string();
}

void ov::intel_gna::rt_info::remove_transpose_fusable(const std::shared_ptr<Node>& node) {
    auto& rt_info = node->get_rt_info();
    rt_info.erase(GNATransposeFusable::get_type_info_static());
}

bool ov::intel_gna::rt_info::is_transpose_fusable(const std::shared_ptr<Node>& node) {
    const auto& rt_info = node->get_rt_info();
    return rt_info.count(GNATransposeFusable::get_type_info_static());
}
