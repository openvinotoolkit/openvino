// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/rt_info/reduceop_path.hpp"

void ov::mark_reduceop_path(const std::shared_ptr<Node>& node) {
    node->get_rt_info().emplace(ReduceOpPath::get_type_info_static(), ReduceOpPath{});
}

bool ov::is_reduceop_path(const std::shared_ptr<const Node>& node) {
    return node->get_rt_info().count(ReduceOpPath::get_type_info_static());
}
