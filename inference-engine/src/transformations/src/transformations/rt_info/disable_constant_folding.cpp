// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/rt_info/disable_constant_folding.hpp"

void ov::disable_constant_folding(const std::shared_ptr<Node>& node) {
    auto & rt_info = node->get_rt_info();
    rt_info[DisableConstantFolding::get_type_info_static()] = std::make_shared<DisableConstantFolding>(true);
}

void ov::enable_constant_folding(const std::shared_ptr<Node>& node) {
    auto & rt_info = node->get_rt_info();
    rt_info.erase(DisableConstantFolding::get_type_info_static());
}

bool ov::constant_folding_is_disabled(const std::shared_ptr<Node> &node) {
    const auto & rt_info = node->get_rt_info();
    return rt_info.count(DisableConstantFolding::get_type_info_static());
}
