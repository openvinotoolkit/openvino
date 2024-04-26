// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/rt_info/constant_node.hpp"

void ov::mark_as_constant_node(const std::shared_ptr<Node>& node) {
    auto& rt_info = node->get_rt_info();
    rt_info[ConstantNode::get_type_info_static()] = ConstantNode();
}

void ov::mark_as_constant_node(const Output<Node>& node) {
    ov::mark_as_constant_node(node.get_node_shared_ptr());
}

bool ov::is_marked_as_constant_node(const std::shared_ptr<Node>& node) {
    const auto& rt_info = node->get_rt_info();
    return rt_info.find(ConstantNode::get_type_info_static()) != rt_info.end();
}

bool ov::is_marked_as_constant_node(const Output<Node>& node) {
    return ov::is_marked_as_constant_node(node.get_node_shared_ptr());
}
