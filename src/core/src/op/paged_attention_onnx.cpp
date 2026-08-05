// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_attention_onnx.hpp"

#include "itt.hpp"

namespace ov::op::internal {

PagedAttention::PagedAttention(const OutputVector& args,
                               int64_t num_heads,
                               int64_t kv_num_heads,
                               float scale,
                               float softcap,
                               int64_t local_window_size,
                               bool do_rotary,
                               bool rotary_interleaved)
    : Op(args),
      m_num_heads(num_heads),
      m_kv_num_heads(kv_num_heads),
      m_scale(scale),
      m_softcap(softcap),
      m_local_window_size(local_window_size),
      m_do_rotary(do_rotary),
      m_rotary_interleaved(rotary_interleaved) {
    constructor_validate_and_infer_types();
}

void PagedAttention::validate_and_infer_types() {
    OV_OP_SCOPE(PagedAttention_validate_and_infer_types);
    // PagedAttention (ONNX Runtime com.microsoft) expects the following inputs (the packed-QKV form is split
    // into separate Q/K/V by the frontend, so this op always receives three separate 2-D activation tensors):
    //   0 query, 1 key, 2 value                    (2-D [num_tokens, heads * head_size])
    //   3 key_cache, 4 value_cache                 (4-D [num_blocks, block_size, kv_num_heads, head_size])
    //   5 cumulative_sequence_length               (1-D [batch_size + 1], i32)
    //   6 past_seqlens                             (1-D [batch_size], i32)
    //   7 block_table                              (2-D [batch_size, max_blocks_per_sequence], i32)
    //   8 cos_cache, 9 sin_cache                   (2-D, present only when do_rotary)
    // and produces: 0 output [num_tokens, num_heads * head_size], 1 key_cache_out, 2 value_cache_out (the
    // in-place-updated caches, same shape/type as inputs 3/4 - honoring the ONNX cache-in -> cache-out contract).
    const auto input_size = get_input_size();
    NODE_VALIDATION_CHECK(this,
                          input_size == 8 || input_size == 10,
                          "PagedAttention expects 8 inputs (or 10 when do_rotary adds cos_cache/sin_cache), got ",
                          input_size);
    NODE_VALIDATION_CHECK(this,
                          !m_do_rotary || input_size == 10,
                          "PagedAttention with do_rotary enabled requires cos_cache (input 8) and sin_cache "
                          "(input 9).");

    // Query/activation is float; the attention math is computed in float precision. The ONNX spec constrains
    // the type to {f16, bf16}; f32 is additionally accepted for unit-test convenience (the decomposition math
    // is identical). This matches the ScaledDotProductAttention the op lowers to.
    const auto& q_type = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          q_type == element::f32 || q_type == element::f16 || q_type == element::bf16,
                          "PagedAttention supports the following query element types: {f32, f16, bf16}");

    NODE_VALIDATION_CHECK(this,
                          m_kv_num_heads > 0 && m_num_heads % m_kv_num_heads == 0,
                          "PagedAttention requires kv_num_heads > 0 and num_heads divisible by kv_num_heads, got "
                          "num_heads = ",
                          m_num_heads,
                          ", kv_num_heads = ",
                          m_kv_num_heads);

    // local_window_size == -1 disables the window; a value >= 1 selects a sliding window. A window of size 0 is
    // an empty attention (every query masks all keys, including its own diagonal) and is not a valid config; it
    // is rejected here (and in the frontend) so the finite-lowest() mask stays NaN-safe (every row keeps its
    // own diagonal key).
    NODE_VALIDATION_CHECK(this,
                          m_local_window_size == -1 || m_local_window_size >= 1,
                          "PagedAttention: local_window_size must be -1 (disabled) or >= 1, got ",
                          m_local_window_size);

    // Any batch size is supported: a statically-known batch == 1 takes the single-sequence fast path, and
    // everything else (static batch > 1, or a dynamic batch) takes the general variable-length path where each
    // packed token is mapped by its own sequence's past length and block_table row.

    // output[0] has the same [num_tokens, num_heads * head_size] shape and float type as the (separate) query.
    set_output_type(0, q_type, get_input_partial_shape(0));
    // key_cache_out / value_cache_out are the in-place-updated caches: same shape and element type as inputs 3/4.
    set_output_type(1, get_input_element_type(3), get_input_partial_shape(3));
    set_output_type(2, get_input_element_type(4), get_input_partial_shape(4));
}

bool PagedAttention::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(PagedAttention_visit_attributes);
    visitor.on_attribute("num_heads", m_num_heads);
    visitor.on_attribute("kv_num_heads", m_kv_num_heads);
    visitor.on_attribute("scale", m_scale);
    visitor.on_attribute("softcap", m_softcap);
    visitor.on_attribute("local_window_size", m_local_window_size);
    visitor.on_attribute("do_rotary", m_do_rotary);
    visitor.on_attribute("rotary_interleaved", m_rotary_interleaved);
    return true;
}

std::shared_ptr<ov::Node> PagedAttention::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OV_OP_SCOPE(PagedAttention_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<PagedAttention>(new_args,
                                            m_num_heads,
                                            m_kv_num_heads,
                                            m_scale,
                                            m_softcap,
                                            m_local_window_size,
                                            m_do_rotary,
                                            m_rotary_interleaved);
}

}  // namespace ov::op::internal
