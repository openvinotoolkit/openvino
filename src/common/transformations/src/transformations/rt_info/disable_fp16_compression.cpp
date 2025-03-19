// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/rt_info/disable_fp16_compression.hpp"

namespace {
const std::string& get_postponed_fp16_compression_tag() {
    static const std::string postponed_fp16_compression_tag("postponed_fp16_compression");
    return postponed_fp16_compression_tag;
}
}  // namespace

void ov::disable_fp16_compression(const std::shared_ptr<Node>& node) {
    auto& rt_info = node->get_rt_info();
    rt_info[DisableFP16Compression::get_type_info_static()] = DisableFP16Compression{};
}

void ov::enable_fp16_compression(const std::shared_ptr<Node>& node) {
    auto& rt_info = node->get_rt_info();
    rt_info.erase(DisableFP16Compression::get_type_info_static());
}

bool ov::fp16_compression_is_disabled(const std::shared_ptr<const Node>& node) {
    const auto& rt_info = node->get_rt_info();
    return rt_info.count(DisableFP16Compression::get_type_info_static());
}

void ov::postpone_fp16_compression(ov::RTMap& rt_info) {
    rt_info[get_postponed_fp16_compression_tag()] = true;
}

bool ov::is_fp16_compression_postponed(const ov::RTMap& rt_info) {
    return rt_info.count(get_postponed_fp16_compression_tag());
}

void ov::do_not_postpone_fp16_compression(ov::RTMap& rt_info) {
    rt_info.erase(get_postponed_fp16_compression_tag());
}
