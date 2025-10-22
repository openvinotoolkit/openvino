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
};

}   // namespace ov::intel_gpu
