// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

class DynamicQuantizeGatedMLP : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("DynamicQuantizeGatedMLP");
    DynamicQuantizeGatedMLP(uint64_t group_size, bool asymmetric = false, bool precomputed_reduction = true, bool use_gs128_for_int8_per_token = false, bool use_gs128_for_linear_attention = false);
    static bool ShouldUseGs128(bool is_wei_i8u8, bool use_gs128_for_int8_per_token, uint64_t group_size, bool use_gs128_for_linear_attention) {
        return group_size == UINT64_MAX && (use_gs128_for_linear_attention || (is_wei_i8u8 && use_gs128_for_int8_per_token));
    }
};

}  // namespace ov::intel_gpu
