// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

/// Converts a decomposed-attention Select (where) mask into the additive form that the common
/// ov::pass::SDPAFusion already understands, without modifying the shared common transformations:
///     Select(mask, scores, sentinel)  ->  scores + Select(mask, 0, mask_value)
///
/// The rewrite is only applied when all of the following hold:
/// - the scores element type is f16 or bf16 (the fused SDPA primitive supports these only);
/// - the Select uses NUMPY broadcasting and has a single consumer;
/// - the complete decomposed attention is matched - MatMul(Q,K) -> [scale] -> Select ->
///   [Reshape] -> Softmax -> [Reshape] -> MatMul(probs,V) - so unrelated Select -> Softmax
///   graphs are left untouched;
/// - the attention Softmax is a v8 Softmax on the last axis with a single consumer, the MatMuls
///   are plain (no transposed A) and the scores rank is at most 4 - the hard requirements of the
///   common ov::pass::SDPAFusion, so the inserted Add is guaranteed to be fused;
/// - the masked-out value is a scalar constant of -65504 or lower (a true -inf, or one saturated
///   to the lowest f16 value).
///
/// mask_value is half of the lowest value of the scores element type: a true -inf cannot be built
/// for f16/bf16 constants, and this keeps a fully masked row finite. Once normalized by the
/// following Softmax, masked entries underflow to 0 for realistic attention score ranges; the
/// remaining equivalence caveats are documented in the .cpp.
class SDPASelectMaskFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("SDPASelectMaskFusion");
    SDPASelectMaskFusion();
};

}   // namespace ov::intel_gpu
