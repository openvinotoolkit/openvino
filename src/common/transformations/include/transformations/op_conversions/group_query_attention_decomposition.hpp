// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/group_query_attention.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API GroupQueryAttentionDecomposition;

}  // namespace pass
}  // namespace ov

class ov::pass::GroupQueryAttentionDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("GroupQueryAttentionDecomposition");
    GroupQueryAttentionDecomposition();

protected:
    ov::OutputVector decompose(std::shared_ptr<ov::op::internal::GroupQueryAttention> node);
    virtual std::shared_ptr<ov::Node> make_sdpa(const ov::Output<ov::Node>& query,
                                                const ov::Output<ov::Node>& key,
                                                const ov::Output<ov::Node>& value,
                                                const ov::Output<ov::Node>& mask,
                                                const ov::Output<ov::Node>& scale,
                                                const ov::Output<ov::Node>& sink,
                                                bool is_causal);
    std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<op::v3::ShapeOf>& shape,
                                             const std::vector<int>& dims);
    std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<ov::Node>& node, const std::vector<int>& dims);
    std::shared_ptr<ov::Node> rotaryEmbedding(ov::Output<ov::Node> input,
                                              ov::Output<ov::Node> cos,
                                              ov::Output<ov::Node> sin,
                                              bool interleaved);
    // Resident row count of a windowed KV cache for a given absolute sequence length, as an i64 scalar node:
    // end(x) = x <= capacity ? x : x - gap * ceil((x - capacity) / gap), with gap = capacity - window + 1.
    // Mirrors ONNX Runtime's WindowedCacheEnd; blocks of `gap` rows are reclaimed at once on front eviction.
    std::shared_ptr<ov::Node> windowed_cache_end(const ov::Output<ov::Node>& seqlen_scalar,
                                                 const ov::Output<ov::Node>& capacity_scalar,
                                                 int64_t local_window_size);

    // Additive float attention mask for SDPA. When causal is true: causal mask, plus an optional sliding-window
    // band (local_window_size >= 1) masking keys older than the window. When causal is false (bidirectional):
    // no query-relative masking; only the unused cache tail beyond total_sequence_length (past + current) is
    // masked (local_window_size is always -1 in this case, enforced upstream). Either way the result is
    // optionally fused with an external bias. Masked positions use the compute type's finite lowest() so a
    // fully-masked row cannot softmax to NaN.
    virtual std::shared_ptr<ov::Node> make_attention_mask(const ov::Output<ov::Node>& curr_seqlen_scalar,
                                                          const ov::Output<ov::Node>& kv_len_scalar,
                                                          const ov::Output<ov::Node>& kv_len_1d,
                                                          const ov::Output<ov::Node>& past_seqlen,
                                                          const ov::element::Type& compute_type,
                                                          bool causal,
                                                          int64_t local_window_size,
                                                          const ov::Output<ov::Node>& external_bias,
                                                          const ov::Output<ov::Node>& bias_col_offset,
                                                          bool sliding_window_cache,
                                                          float scale,
                                                          bool has_sink);
    // Reshape a flat KV-cache dequant scale so it broadcasts against a [B, kv_num_heads, S, head_size] tensor:
    // PER_CHANNEL -> [1, kv_num_heads, 1, head_size]; PER_TENSOR -> [1, 1, 1, 1].
    std::shared_ptr<ov::Node> make_kv_scale(const ov::Output<ov::Node>& scale,
                                            int64_t kv_num_heads,
                                            ov::op::internal::GroupQueryAttentionQuantType quant_type);
    // Dequantize a quantized (i8/u8/f8e4m3) KV cache to compute_type: x = q * scale (symmetric, zero point = 0).
    std::shared_ptr<ov::Node> dequantize_kv(const ov::Output<ov::Node>& quantized,
                                            const ov::Output<ov::Node>& scale,
                                            int64_t kv_num_heads,
                                            int64_t kv_cache_bit_width,
                                            ov::op::internal::GroupQueryAttentionQuantType quant_type,
                                            const ov::element::Type& compute_type);
    // Quantize current float KV tokens into the cache type: integer i8/u8 uses clamp(round(x / scale))
    // (round-half-to-even); f8e4m3 uses clamp(x / scale, +/-448) then Convert (no integer round).
    std::shared_ptr<ov::Node> quantize_kv(const ov::Output<ov::Node>& current,
                                          const ov::Output<ov::Node>& scale,
                                          int64_t kv_num_heads,
                                          int64_t kv_cache_bit_width,
                                          ov::op::internal::GroupQueryAttentionQuantType quant_type,
                                          const ov::element::Type& cache_type);
};
