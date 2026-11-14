// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {
class TRANSFORMATIONS_API NormalizeVLLMRoPE;
}  // namespace ov::pass

/**
 * @ingroup ov_transformation_common_api
 * @brief Rewrites vLLM's neox-style RoPE lowering into the rotate-half form RoPEFusionGPTNEOX matches.
 *
 * vLLM rotates the two halves of x separately with half-width cos/sin tables, which no RoPE fusion matches:
 *
 * Before:                                        After:
 *
 *   x1, x2 = Split(x, axis=-1)                     x1, x2 = VariadicSplit(x, axis=-1, [D/2, D/2])
 *   Concat(x1 * cos + (x2 * sin) * -1,             x * Concat(cos, cos)
 *          x2 * cos + x1 * sin, axis=-1)             + Concat(x2 * -1, x1, axis=-1) * Concat(sin, sin)
 *
 * Applied when:
 *  - one 2-way Split or VariadicSplit cuts the last dimension of x into equal halves that feed the Multiplies directly;
 *  - cos and sin have the same shape, with the halves' static width D/2 as their last dimension;
 *  - the Concat joins the two branches along the last axis.
 *
 * The subtraction must already be a Multiply(-1) + Add, so ov::pass::ConvertSubtract has to run first.
 * CommonOptimizations registers the pass only for models with the "vllm_model" rt_info flag.
 */
class ov::pass::NormalizeVLLMRoPE : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("NormalizeVLLMRoPE");
    NormalizeVLLMRoPE();
};
