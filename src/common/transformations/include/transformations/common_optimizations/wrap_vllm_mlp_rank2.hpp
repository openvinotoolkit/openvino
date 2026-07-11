// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {

// Wrap rank-2 vLLM MLP block with Unsqueeze(axis=0) at the shared
// input and Squeeze(axis=0) at the down_proj output so intel_cpu's
// rank-3-only LLMMLPFusion can match unchanged. Must run after
// NormalizeVLLMMLP (which produces the VariadicSplit form) and before
// MLPFusionPass.
class TRANSFORMATIONS_API WrapVLLMMLPRank2 : public MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("WrapVLLMMLPRank2");
    WrapVLLMMLPRank2();
};

}  // namespace ov::pass
