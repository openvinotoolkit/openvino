// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/rt_info/dequantization_node.hpp"

void ov::mark_as_dequantization_node(const std::shared_ptr<Node>& node) {
    auto& rt_info = node->get_rt_info();
    rt_info[DequantizationNode::get_type_info_static()] = DequantizationNode();
}

void ov::unmark_dequantization_node(const std::shared_ptr<Node>& node) {
    node->get_rt_info().erase(DequantizationNode::get_type_info_static());
}

bool ov::is_dequantization_node(const std::shared_ptr<const Node>& node) {
    const auto& rt_info = node->get_rt_info();
    return rt_info.find(DequantizationNode::get_type_info_static()) != rt_info.end();
}
