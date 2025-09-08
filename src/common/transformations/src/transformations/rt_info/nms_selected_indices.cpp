// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/rt_info/nms_selected_indices.hpp"

void ov::set_nms_selected_indices(Node* node) {
    node->get_rt_info().emplace(NmsSelectedIndices::get_type_info_static(), NmsSelectedIndices{});
}

bool ov::has_nms_selected_indices(const Node* node) {
    return node->get_rt_info().count(NmsSelectedIndices::get_type_info_static());
}
