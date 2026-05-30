// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

class DynamicQuantizeGatedMLP : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("DynamicQuantizeGatedMLP");
    DynamicQuantizeGatedMLP(uint64_t group_size, bool asymmetric = false, bool precomputed_reduction = true);
};

}  // namespace ov::intel_gpu
