// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/core/visibility.hpp"
#include "openvino/op/normalize_l2.hpp"

namespace ov::intel_gpu {

bool normalize_l2_supports_across_x(const ov::op::v0::NormalizeL2& op);

/**
 * @brief Decomposes NormalizeL2 into subgraph
 */
class NormalizeL2Decomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("NormalizeL2DecompositionGPU");
    NormalizeL2Decomposition(bool use_fp32_reducesum = false);
};

}  // namespace ov::intel_gpu
