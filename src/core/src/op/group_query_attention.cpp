// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/group_query_attention.hpp"

#include "itt.hpp"

namespace ov::op::internal {

GroupQueryAttention::GroupQueryAttention(const OutputVector& args,
                                         int64_t num_heads,
                                         int64_t kv_num_heads,
                                         float scale,
                                         bool do_rotary,
                                         bool rotary_interleaved,
                                         int64_t kv_cache_bit_width,
                                         const std::string& k_quant_type,
                                         const std::string& v_quant_type)
    : Op(args),
      m_num_heads(num_heads),
      m_kv_num_heads(kv_num_heads),
      m_scale(scale),
      m_do_rotary(do_rotary),
      m_rotary_interleaved(rotary_interleaved),
      m_kv_cache_bit_width(kv_cache_bit_width),
      m_k_quant_type(k_quant_type),
      m_v_quant_type(v_quant_type) {
    constructor_validate_and_infer_types();
}

void GroupQueryAttention::validate_and_infer_types() {
    OV_OP_SCOPE(GroupQueryAttention_validate_and_infer_types);
    // GroupQueryAttention expects the following inputs:
    // query, key, value, past_key, past_value, seqlens_k, total_sequence_length.
    // In the rotary embedding case, following inputs are also required: cos_cache, sin_cache.
    // All qkv tensors should have the shape [batch, num_heads, seq_len, head_size] ([B, N, S, H]).
    // The operation produces three outputs:
    // 1. Output tensor of shape [B, S, N * H].
    // 2. Present_key tensor of shape [B, N, S, H].
    // 3. Present_value tensor of shape [B, N, S, H].
    // Note: seqlens_k represents the number of 1's in the attention_mask minus 1.

    const auto& q_shape = get_input_partial_shape(0);
    const auto& batch_size = q_shape[0];
    const auto& sequence_len = q_shape[2];
    const auto& head_size = q_shape[3];
    // present_key/present_value keep the past KV layout: kv head size may differ from the query head size
    // when the cache is quantized (e.g. 4-bit values packed two per byte), so take it from past_key (input 3).
    const auto& past_kv_shape = get_input_partial_shape(3);
    auto kv_shape = PartialShape{batch_size, m_kv_num_heads, past_kv_shape[2], past_kv_shape[3]};
    auto& output_kv_len = kv_shape[2];

    if (output_kv_len.is_dynamic() || sequence_len.is_dynamic()) {
        // For dynamic shapes, concatenate the past and current sequence lengths.
        output_kv_len += sequence_len;
    }

    // Query/activation (input 0) is always float; attention itself is computed in float precision.
    const auto& q_type = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          q_type == element::f32 || q_type == element::f16,
                          "GroupQueryAttention supports following query element types: {f32, f16}");

    // The KV cache (past_key/past_value, input 3/4) may be quantized. present_key/present_value inherit the
    // cache element type so a quantized (i8/u8) cache round-trips from past to present, matching the ONNX spec.
    const auto& kv_cache_type = get_input_element_type(3);
    NODE_VALIDATION_CHECK(this,
                          kv_cache_type == element::f32 || kv_cache_type == element::f16 ||
                              kv_cache_type == element::i8 || kv_cache_type == element::u8,
                          "GroupQueryAttention supports following KV cache element types: {f32, f16, i8, u8}");

    if (is_kv_quantized()) {
        // Quantized KV cache: i8 (8-bit) or u8 (4-bit values packed two per byte). Requires float dequant scales.
        NODE_VALIDATION_CHECK(this,
                              kv_cache_type == element::i8 || kv_cache_type == element::u8,
                              "GroupQueryAttention with quantized KV cache requires an i8 or u8 past/present KV type");
        NODE_VALIDATION_CHECK(this,
                              m_kv_cache_bit_width == 8 || m_kv_cache_bit_width == 4,
                              "GroupQueryAttention supports kv_cache_bit_width of 8 or 4, got: ",
                              m_kv_cache_bit_width);
        NODE_VALIDATION_CHECK(this,
                              m_k_quant_type == m_v_quant_type,
                              "GroupQueryAttention requires matching k_quant_type and v_quant_type, got: ",
                              m_k_quant_type,
                              " and ",
                              m_v_quant_type);
        NODE_VALIDATION_CHECK(this,
                              get_input_size() > 13,
                              "GroupQueryAttention with quantized KV cache requires k_scale (input 12) and "
                              "v_scale (input 13)");
        NODE_VALIDATION_CHECK(this,
                              get_input_element_type(12) == element::f32 && get_input_element_type(13) == element::f32,
                              "GroupQueryAttention k_scale/v_scale must be f32");
    }

    set_output_type(0, q_type, PartialShape{batch_size, sequence_len, head_size * m_num_heads});
    for (auto&& port : {1, 2}) {
        set_output_type(port, kv_cache_type, kv_shape);
    }
}

bool GroupQueryAttention::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(GroupQueryAttention_visit_attributes);
    visitor.on_attribute("do_rotary", m_do_rotary);
    visitor.on_attribute("k_quant_type", m_k_quant_type);
    visitor.on_attribute("kv_cache_bit_width", m_kv_cache_bit_width);
    visitor.on_attribute("kv_num_heads", m_kv_num_heads);
    visitor.on_attribute("num_heads", m_num_heads);
    visitor.on_attribute("rotary_interleaved", m_rotary_interleaved);
    visitor.on_attribute("scale", m_scale);
    visitor.on_attribute("v_quant_type", m_v_quant_type);
    return true;
}

std::shared_ptr<ov::Node> GroupQueryAttention::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OV_OP_SCOPE(GroupQueryAttention_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<GroupQueryAttention>(new_args,
                                                 m_num_heads,
                                                 m_kv_num_heads,
                                                 m_scale,
                                                 m_do_rotary,
                                                 m_rotary_interleaved,
                                                 m_kv_cache_bit_width,
                                                 m_k_quant_type,
                                                 m_v_quant_type);
}

}  // namespace ov::op::internal
