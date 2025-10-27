// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

class DynamicQuantizeFullyConnected: public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("DynamicQuantizeFullyConnected");
    DynamicQuantizeFullyConnected(uint64_t group_size, bool asymmetric = false, bool precompute_sum = true, bool use_gs128_for_int8_per_token = false);
    static bool ShouldUseGs128(uint64_t is_wei_i8u8, bool use_gs128_for_int8_per_token, uint64_t group_size) {
        return (is_wei_i8u8 && use_gs128_for_int8_per_token && group_size == UINT64_MAX);
    }
};

}   // namespace ov::intel_gpu
