// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gna_node_id.hpp"

void ov::intel_gna::rt_info::set_node_id(const std::shared_ptr<Node>& node, uint64_t id) {
    auto& rt_info = node->get_rt_info();
    rt_info[GnaNodeId::get_type_info_static()] = id;
}

void ov::intel_gna::rt_info::remove_node_id(const std::shared_ptr<Node>& node) {
    auto& rt_info = node->get_rt_info();
    rt_info.erase(GnaNodeId::get_type_info_static());
}

uint64_t ov::intel_gna::rt_info::get_node_id(const std::shared_ptr<Node>& node) {
    const auto& rt_info = node->get_rt_info();
    return rt_info.at(GnaNodeId::get_type_info_static()).as<uint64_t>();
}
