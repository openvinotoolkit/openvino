// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

/// Converts a decomposed-attention Select (where) mask into the additive form that the
/// common ov::pass::SDPAFusion already understands, without modifying the shared common
/// transformations:
///     Select(mask, scores, neg_inf)  ==  scores + Select(mask, 0, neg_inf)
/// (valid because the following Softmax normalizes the masked-out entries to 0).
///
/// Run before CommonOptimizations so the emitted Add(scores, mask) -> Softmax -> MatMul
/// pattern is picked up by ov::pass::SDPAFusion (registered inside MOCTransformations) and
/// fused into a v13 ScaledDotProductAttention compiled to the ocl::sdpa kernels.
class SDPASelectMaskFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("SDPASelectMaskFusion");
    SDPASelectMaskFusion();
};

}   // namespace ov::intel_gpu
