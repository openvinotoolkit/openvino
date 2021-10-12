// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/rt_info/nms_selected_indices.hpp"

template class ov::VariantImpl<ov::NmsSelectedIndices>;

void ov::set_nms_selected_indices(Node * node) {
    auto & rt_info = node->get_rt_info();
    rt_info[NmsSelectedIndices::get_type_info_static()] = std::make_shared<NmsSelectedIndices>(true);
}

bool ov::has_nms_selected_indices(const Node * node) {
    const auto & rt_info = node->get_rt_info();
    return rt_info.count(NmsSelectedIndices::get_type_info_static());
}
