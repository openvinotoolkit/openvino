// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/paged_attention_onnx.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API PagedAttentionDecomposition;

}  // namespace pass
}  // namespace ov

// Decomposes the internal ov::op::internal::PagedAttention (the ONNX com.microsoft.PagedAttention op) into a
// ScaledDotProductAttention-based subgraph that honors the ONNX cache-in -> cache-out contract, so a standalone
// ONNX model runs on CPU/GPU. This is the "decompose by default" half of the design; a plugin/serving pipeline
// can disable this pass (transformation_callback) to keep the op native. Single-sequence (batch_size == 1) for
// now; multi-sequence varlen is added later.
class ov::pass::PagedAttentionDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PagedAttentionDecomposition");
    PagedAttentionDecomposition();

private:
    ov::OutputVector decompose(std::shared_ptr<ov::op::internal::PagedAttention> node);

    // Gather one or more dimensions of a node's shape as an i64 1-D tensor (Gather on ShapeOf).
    std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<op::v3::ShapeOf>& shape,
                                             const std::vector<int>& dims);
    std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<ov::Node>& node, const std::vector<int>& dims);

    // Full RoPE (interleaved and non-interleaved), applied to a [1, heads, tokens, head_size] tensor.
    std::shared_ptr<ov::Node> rotaryEmbedding(ov::Output<ov::Node> input,
                                              ov::Output<ov::Node> cos,
                                              ov::Output<ov::Node> sin,
                                              bool interleaved);

    // Physical flat slot index (i32, shape [count]) into a cache reshaped to [num_blocks * block_size, ...] for
    // `count` logical positions starting at `start_pos`: slot(p) = block_table[p / block_size] * block_size +
    // p % block_size. Mirrors the ONNX Runtime / OV PagedAttention paged KV-cache indexing.
    std::shared_ptr<ov::Node> build_slot_indices(const ov::Output<ov::Node>& block_table_row,
                                                 const ov::Output<ov::Node>& start_pos_scalar,
                                                 const ov::Output<ov::Node>& count_scalar,
                                                 const ov::Output<ov::Node>& block_size_scalar);

    // Additive float attention mask [curr, kv_len] for SDPA: causal (key j masked when j > past + i), optionally
    // fused with a sliding-window band (masked when (past + i) - j >= local_window_size). Masked positions use
    // the compute type's finite lowest() (f16/bf16/f32 branches) so a fully-masked row cannot softmax to NaN.
    std::shared_ptr<ov::Node> make_attention_mask(const ov::Output<ov::Node>& curr_seqlen_scalar,
                                                  const ov::Output<ov::Node>& kv_len_scalar,
                                                  const ov::Output<ov::Node>& past_seqlen_scalar,
                                                  const ov::element::Type& compute_type,
                                                  int64_t local_window_size);
};
