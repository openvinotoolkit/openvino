// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/rt_info/keep_fp16_const.hpp"

void ov::enable_keep_fp16_const(const std::shared_ptr<Node>& node) {
    auto& rt_info = node->get_rt_info();
    rt_info[KeepFP16Const::get_type_info_static()] = KeepFP16Const{};
}

void ov::disable_keep_fp16_const(const std::shared_ptr<Node>& node) {
    auto& rt_info = node->get_rt_info();
    rt_info.erase(KeepFP16Const::get_type_info_static());
}

bool ov::is_keep_fp16_const(const std::shared_ptr<const Node>& node) {
    const auto& rt_info = node->get_rt_info();
    return rt_info.count(KeepFP16Const::get_type_info_static());
}
