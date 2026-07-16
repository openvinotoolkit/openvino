// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_gpu {

/// Captures the ggml-openvino frontend's top-k expert matmul, expressed with public ops
/// as the rank-2 carrier
///   Reshape( [Convert]( Gather(CompressedWeightsBlock[n_expert, m*k], ids[n_tokens, n_used]) ) )
///     -> MatMul(activations, ., transpose_b),
/// and rewrites it to the internal ov::op::internal::GatherMatmul (rebuilding the compressed
/// weight block at the rank-3 [n_expert, N, K] shape GatherMatmul needs) so the downstream
/// ConvertGatherMatmulToGatherMatmulCompressed can fold it into GatherMatmulCompressed
/// (gather + dequantize-selected-experts + matmul in one op, weights kept compressed).
///
/// The frontend emits only public ops; this plugin pass does the internal-op conversion.
class ConvertGgmlMoeGatherToGatherMatmul : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertGgmlMoeGatherToGatherMatmul");
    ConvertGgmlMoeGatherToGatherMatmul();

private:
    // The CompressedWeightsBlock pattern node, kept so the callback can query its anchors.
    std::shared_ptr<ov::Node> m_weights_block;
};

}  // namespace ov::intel_gpu
