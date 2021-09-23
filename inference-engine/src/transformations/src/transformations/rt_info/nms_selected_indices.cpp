// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/rt_info/nms_selected_indices.hpp"

template class ov::VariantImpl<ov::NmsSelectedIndices>;

constexpr ov::VariantTypeInfo ov::VariantWrapper<ov::NmsSelectedIndices>::type_info;

void ov::set_nms_selected_indices(Node * node) {
    auto & rt_info = node->get_rt_info();
    rt_info[VariantWrapper<NmsSelectedIndices>::type_info] = make_variant<NmsSelectedIndices>({});
}

bool ov::has_nms_selected_indices(const Node * node) {
    const auto & rt_info = node->get_rt_info();
    return rt_info.count(VariantWrapper<NmsSelectedIndices>::type_info);
}
