// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/shape_of_subgraph_root_attribute.hpp"

void ov::mark_as_shape_of_subgraph_root(const std::shared_ptr<Node>& node) {
    auto& rt_info = node->get_rt_info();
    rt_info[ShapeOfSubgraphRoot::get_type_info_static()] = ShapeOfSubgraphRoot{};
}

bool ov::is_shape_of_subgraph_root(const std::shared_ptr<const Node>& node) {
    const auto& rt_info = node->get_rt_info();
    return rt_info.count(ShapeOfSubgraphRoot::get_type_info_static()) != 0;
}
