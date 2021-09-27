// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/rt_info/disable_constant_folding.hpp"

template class ov::VariantImpl<ov::DisableConstantFolding>;

void ov::disable_constant_folding(const std::shared_ptr<Node>& node) {
    auto & rt_info = node->get_rt_info();
    rt_info[VariantWrapper<DisableConstantFolding>::get_type_info_static()] = make_variant<DisableConstantFolding>({});
}

void ov::enable_constant_folding(const std::shared_ptr<Node>& node) {
    auto & rt_info = node->get_rt_info();
    rt_info.erase(VariantWrapper<DisableConstantFolding>::get_type_info_static());
}

bool ov::constant_folding_is_disabled(const std::shared_ptr<Node> &node) {
    const auto & rt_info = node->get_rt_info();
    return rt_info.count(VariantWrapper<DisableConstantFolding>::get_type_info_static());
}
