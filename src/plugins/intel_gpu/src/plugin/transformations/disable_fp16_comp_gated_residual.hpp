// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

/**
 * @brief Keeps Qwen-Image gated residual paths in FP32 when they feed normalization.
 *
 * The branch MatMul or Multiply result in Add(residual, Multiply(gate, branch))
 * can exceed the FP16 range even when the earlier inputs are representable in FP16.
 * Qwen-Image gates are identified by their nested VariadicSplit producer chain.
 */
class DisableFP16CompForQwenImageGatedResidualPattern : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("DisableFP16CompForQwenImageGatedResidualPattern");
    DisableFP16CompForQwenImageGatedResidualPattern();
};

}  // namespace ov::intel_gpu
