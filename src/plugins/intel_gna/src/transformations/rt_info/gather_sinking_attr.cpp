// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_sinking_attr.hpp"

void ov::intel_gna::rt_info::mark_as_no_gather_sinking_node(const std::shared_ptr<Node>& node) {
    auto& rt_info = node->get_rt_info();
    rt_info[NoGatherSinkingAttr::get_type_info_static()] = NoGatherSinkingAttr();
}

template <typename NodePtr>
bool is_gather_sinking_node_private(NodePtr node) {
    const auto& rt_info = node->get_rt_info();
    return rt_info.find(ov::intel_gna::rt_info::NoGatherSinkingAttr::get_type_info_static()) == rt_info.end();
}

bool ov::intel_gna::rt_info::is_gather_sinking_node(const std::shared_ptr<Node>& node) {
    return is_gather_sinking_node_private(node);
}

bool ov::intel_gna::rt_info::is_gather_sinking_node(const Node* node) {
    return is_gather_sinking_node_private(node);
}

bool ov::intel_gna::rt_info::is_gather_sinking_node(Output<Node> output) {
    return is_gather_sinking_node(output.get_node());
}
