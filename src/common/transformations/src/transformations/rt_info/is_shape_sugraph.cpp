// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/rt_info/is_shape_subgraph.hpp"

void ov::mark_shape_subgraph(const std::shared_ptr<Node>& node) {
    auto& rt_info = node->get_rt_info();
    rt_info[ShapeSubgraph::get_type_info_static()] = ShapeSubgraph{};
}

void ov::unmark_shape_subgraph(const std::shared_ptr<Node>& node) {
    auto& rt_info = node->get_rt_info();
    rt_info.erase(ShapeSubgraph::get_type_info_static());
}

bool ov::is_shape_subgraph(const std::shared_ptr<const Node>& node) {
    const auto& rt_info = node->get_rt_info();
    return rt_info.count(ShapeSubgraph::get_type_info_static());
}
