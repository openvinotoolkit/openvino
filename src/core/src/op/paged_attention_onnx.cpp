// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_attention_onnx.hpp"

#include "itt.hpp"

namespace ov::op::internal {

PagedAttentionONNX::PagedAttentionONNX(const OutputVector& args,
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

void PagedAttentionONNX::validate_and_infer_types() {
    OV_OP_SCOPE(PagedAttentionONNX_validate_and_infer_types);
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

    // Only full-head rotary (rotary_dim == head_size) is supported: the decomposition splits the whole head
    // into two halves sized by cos_cache's last dim (= rotary_dim / 2), so 2 * cos_last_dim must equal head_size.
    // Partial rotary (rotary_dim < head_size, e.g. Phi/GPT-NeoX/GPT-J) is a tracked follow-up. Checked only when
    // the shapes are statically known (head_size derives from query's hidden dim / num_heads); a dynamic hidden
    // or cos width is left to run-time and does not reject the model here.
    if (m_do_rotary && input_size == 10) {
        const auto& q_ps = get_input_partial_shape(0);
        const auto& cos_ps = get_input_partial_shape(8);
        if (q_ps.rank().is_static() && q_ps.rank().get_length() == 2 && q_ps[1].is_static() && m_num_heads > 0 &&
            q_ps[1].get_length() % m_num_heads == 0 && cos_ps.rank().is_static() && cos_ps.rank().get_length() == 2 &&
            cos_ps[1].is_static()) {
            const int64_t head_size = q_ps[1].get_length() / m_num_heads;
            const int64_t rotary_dim = 2 * cos_ps[1].get_length();
            NODE_VALIDATION_CHECK(this,
                                  rotary_dim == head_size,
                                  "PagedAttention: only full-head rotary is supported (cos_cache last dim must "
                                  "equal head_size / 2); partial rotary is not implemented. Got head_size = ",
                                  head_size,
                                  ", rotary_dim (2 * cos_cache last dim) = ",
                                  rotary_dim,
                                  ".");
        }
    }

    // Query/activation is float; the attention math is computed in float precision. The ONNX spec constrains
    // the type to {f16, bf16}; f32 is additionally accepted for unit-test convenience (the decomposition math
    // is identical). This matches the ScaledDotProductAttention the op lowers to.
    const auto& q_type = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          q_type == element::f32 || q_type == element::f16 || q_type == element::bf16,
                          "PagedAttention supports the following query element types: {f32, f16, bf16}");

    // Fail fast on the type/rank invariants the decomposition relies on, but only reject what is provably
    // wrong from static info (dynamic ranks/types are left to run-time). key/value (1/2) and the caches (3/4)
    // must match the query float type; the metadata inputs (5/6/7) are i32.
    for (size_t i : {size_t{1}, size_t{2}, size_t{3}, size_t{4}}) {
        const auto& t = get_input_element_type(i);
        NODE_VALIDATION_CHECK(this,
                              t.is_dynamic() || t == q_type,
                              "PagedAttention: input ",
                              i,
                              " (key/value/key_cache/value_cache) must have the same element type as query (",
                              q_type,
                              "), got ",
                              t);
    }
    for (size_t i : {size_t{5}, size_t{6}, size_t{7}}) {
        const auto& t = get_input_element_type(i);
        NODE_VALIDATION_CHECK(this,
                              t.is_dynamic() || t == element::i32,
                              "PagedAttention: input ",
                              i,
                              " (cumulative_sequence_length/past_seqlens/block_table) must be i32, got ",
                              t);
    }
    // Expected ranks: query/key/value 2-D, key_cache/value_cache 4-D, block_table 2-D, cos/sin 2-D when present.
    const auto check_rank = [this](size_t i, int64_t expected, const char* name) {
        const auto& r = get_input_partial_shape(i).rank();
        NODE_VALIDATION_CHECK(this,
                              r.is_dynamic() || r.get_length() == expected,
                              "PagedAttention: input ",
                              i,
                              " (",
                              name,
                              ") must be ",
                              expected,
                              "-D.");
    };
    check_rank(0, 2, "query");
    check_rank(1, 2, "key");
    check_rank(2, 2, "value");
    check_rank(3, 4, "key_cache");
    check_rank(4, 4, "value_cache");
    check_rank(7, 2, "block_table");
    if (input_size == 10) {
        check_rank(8, 2, "cos_cache");
        check_rank(9, 2, "sin_cache");
    }

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

bool PagedAttentionONNX::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(PagedAttentionONNX_visit_attributes);
    visitor.on_attribute("num_heads", m_num_heads);
    visitor.on_attribute("kv_num_heads", m_kv_num_heads);
    visitor.on_attribute("scale", m_scale);
    visitor.on_attribute("softcap", m_softcap);
    visitor.on_attribute("local_window_size", m_local_window_size);
    visitor.on_attribute("do_rotary", m_do_rotary);
    visitor.on_attribute("rotary_interleaved", m_rotary_interleaved);
    return true;
}

std::shared_ptr<ov::Node> PagedAttentionONNX::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OV_OP_SCOPE(PagedAttentionONNX_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<PagedAttentionONNX>(new_args,
                                                m_num_heads,
                                                m_kv_num_heads,
                                                m_scale,
                                                m_softcap,
                                                m_local_window_size,
                                                m_do_rotary,
                                                m_rotary_interleaved);
}

}  // namespace ov::op::internal
