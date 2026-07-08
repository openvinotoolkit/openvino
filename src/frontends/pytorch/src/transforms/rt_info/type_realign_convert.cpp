// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "type_realign_convert.hpp"

void ov::frontend::pytorch::mark_type_realign_convert(const std::shared_ptr<Node>& node) {
    auto& rt_info = node->get_rt_info();
    rt_info[TypeRealignConvert::get_type_info_static()] = TypeRealignConvert{};
}

void ov::frontend::pytorch::unmark_type_realign_convert(const std::shared_ptr<Node>& node) {
    auto& rt_info = node->get_rt_info();
    rt_info.erase(TypeRealignConvert::get_type_info_static());
}

bool ov::frontend::pytorch::is_type_realign_convert(const std::shared_ptr<const Node>& node) {
    const auto& rt_info = node->get_rt_info();
    return rt_info.count(TypeRealignConvert::get_type_info_static()) > 0;
}
