// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/rt_info/transpose_sinking_attr.hpp"

using namespace ov;

void ov::mark_as_no_sinking_node(const std::shared_ptr<Node>& node) {
    auto& rt_info = node->get_rt_info();
    rt_info[NoTransposeSinkingAttr::get_type_info_static()] = NoTransposeSinkingAttr();
}

namespace {
template <typename NodePtr>
bool is_sinking_node_private(NodePtr node) {
    const auto& rt_info = node->get_rt_info();
    return rt_info.find(NoTransposeSinkingAttr::get_type_info_static()) == rt_info.end();
}
}  // namespace

bool ov::is_sinking_node(const std::shared_ptr<Node>& node) {
    return is_sinking_node_private(node);
}

bool ov::is_sinking_node(const Node* node) {
    return is_sinking_node_private(node);
}
