// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/op/op.hpp"

namespace ov::op::internal {

// This is an experimental operation that is implemented in the plugins.
//
// It models the ONNX Runtime `com.microsoft.PagedAttention` contrib operator: block-paged,
// continuous-batching attention with an in-place KV cache. It is distinct from the serving-runtime
// `ov::op::PagedAttentionExtension` op (28 inputs, externally-managed cache Parameters, no cache
// output): this op honors the ONNX cache-in -> cache-out contract and is lowered by a common
// decomposition (PagedAttentionDecomposition) into a ScaledDotProductAttention-based subgraph.
class OPENVINO_API PagedAttentionONNX : public Op {
public:
    OPENVINO_OP("PagedAttentionONNX");

    PagedAttentionONNX() = default;

    // Inputs (matching the ONNX Runtime op; the packed-QKV form is split into separate Q/K/V by the
    // frontend, so this op always receives three separate 2-D activation tensors):
    //   0 query       [num_tokens, num_heads * head_size]
    //   1 key         [num_tokens, kv_num_heads * head_size]
    //   2 value       [num_tokens, kv_num_heads * head_size]
    //   3 key_cache   [num_blocks, block_size, kv_num_heads, head_size]  (updated in place)
    //   4 value_cache [num_blocks, block_size, kv_num_heads, head_size]  (updated in place)
    //   5 cumulative_sequence_length [batch_size + 1], i32  (prefix-sum of new Q tokens per sequence)
    //   6 past_seqlens [batch_size], i32                    (cached length per sequence)
    //   7 block_table  [batch_size, max_blocks_per_sequence], i32
    //   8 cos_cache    [max_total_seqlen, head_size / 2]    (present only when do_rotary; full-head rotary only)
    //   9 sin_cache    [max_total_seqlen, head_size / 2]    (present only when do_rotary; full-head rotary only)
    // Outputs:
    //   0 output        [num_tokens, num_heads * head_size]
    //   1 key_cache_out   same shape/type as key_cache (in-place updated cache)
    //   2 value_cache_out same shape/type as value_cache
    PagedAttentionONNX(const ov::OutputVector& args,
                       int64_t num_heads,
                       int64_t kv_num_heads,
                       float scale,
                       float softcap,
                       int64_t local_window_size,
                       bool do_rotary,
                       bool rotary_interleaved);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    int64_t get_num_heads() const {
        return m_num_heads;
    }
    int64_t get_kv_num_heads() const {
        return m_kv_num_heads;
    }
    float get_scale() const {
        return m_scale;
    }
    // Softcap value for the attention logits (softcap * tanh(scores / softcap)); 0 disables it.
    float get_softcap() const {
        return m_softcap;
    }
    // Mistral-style local (sliding-window) attention. -1 disables the window (pure causal); a value
    // >= 1 limits each query at absolute position q to keys k in [q - local_window_size + 1, q].
    int64_t get_local_window_size() const {
        return m_local_window_size;
    }
    bool get_do_rotary() const {
        return m_do_rotary;
    }
    bool get_rotary_interleaved() const {
        return m_rotary_interleaved;
    }

private:
    int64_t m_num_heads = 0;
    int64_t m_kv_num_heads = 0;
    float m_scale = 0.0f;
    float m_softcap = 0.0f;
    int64_t m_local_window_size = -1;
    bool m_do_rotary = false;
    bool m_rotary_interleaved = false;
};

}  // namespace ov::op::internal
