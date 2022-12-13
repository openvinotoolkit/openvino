// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/rt_info/disable_fp16_compression.hpp"

void ov::disable_fp16_compression(const std::shared_ptr<Node>& node) {
    auto& rt_info = node->get_rt_info();
    rt_info[DisableFP16Compression::get_type_info_static()] = DisableFP16Compression{};
}

void ov::enable_fp16_compression(const std::shared_ptr<Node>& node) {
    auto& rt_info = node->get_rt_info();
    rt_info.erase(DisableFP16Compression::get_type_info_static());
}

bool ov::fp16_compression_is_disabled(const std::shared_ptr<Node>& node) {
    const auto& rt_info = node->get_rt_info();
    return rt_info.count(DisableFP16Compression::get_type_info_static());
}
