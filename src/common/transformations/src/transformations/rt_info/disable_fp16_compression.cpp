// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/rt_info/disable_fp16_compression.hpp"

#include "transformations/rt_info/disable_precision_conversion.hpp"

void ov::disable_fp16_compression(const std::shared_ptr<Node>& node) {
    disable_conversion(node, element::f16);
}

void ov::enable_fp16_compression(const std::shared_ptr<Node>& node) {
    enable_conversion(node, element::f16);
}

bool ov::fp16_compression_is_disabled(const std::shared_ptr<const Node>& node) {
    if (is_conversion_disabled(node, element::f16)) {
        return true;
    }
    const auto& rt_info = node->get_rt_info();
    return rt_info.count(DisableFP16Compression::get_type_info_static()) > 0;
}
