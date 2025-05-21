// Copyright (C) 2018-2025 Intel Corporation
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
                                         bool rotary_interleaved)
    : Op(args),
      m_num_heads(num_heads),
      m_kv_num_heads(kv_num_heads),
      m_scale(scale),
      m_do_rotary(do_rotary),
      m_rotary_interleaved(rotary_interleaved) {
    constructor_validate_and_infer_types();
}

void GroupQueryAttention::validate_and_infer_types() {
    OV_OP_SCOPE(GroupQueryAttention_validate_and_infer_types);
    // GQA expectes the following inputs: query, key, value, past_key, past_value, seqlens_k, cos_cache, sin_cache
    // All qkv's should have the shape [batch, num_heads, seq_len, head_size] ([B, N, S, H])
    // It has three outputs: output of shape [B, S, N * H], and present_key/value of shape [B, N, S, H]
    // seqlens_k is number of 1's in the attention_mask minus 1

    const auto& q_shape = get_input_partial_shape(0);
    const auto& batch_size = q_shape[0];
    const auto& sequence_len = q_shape[2];
    const auto& head_size = q_shape[3];
    const auto& past_sequence_len = get_input_partial_shape(3)[2];
    const auto& output_kv_len = past_sequence_len + sequence_len;

    const auto& element_type = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          element_type == element::f32 || element_type == element::f16,
                          "GroupQueryAttention only suuports f32 and f16");

    set_output_type(0, element_type, PartialShape{batch_size, sequence_len, head_size * m_num_heads});
    set_output_type(1, element_type, PartialShape{batch_size, m_kv_num_heads, output_kv_len, head_size});
    set_output_type(2, element_type, PartialShape{batch_size, m_kv_num_heads, output_kv_len, head_size});
}

bool GroupQueryAttention::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(GroupQueryAttention_visit_attributes);
    visitor.on_attribute("do_rotary", m_do_rotary);
    visitor.on_attribute("kv_num_heads", m_kv_num_heads);
    visitor.on_attribute("num_heads", m_num_heads);
    visitor.on_attribute("rotary_interleaved", m_rotary_interleaved);
    visitor.on_attribute("scale", m_scale);
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
                                                 m_rotary_interleaved);
}

}  // namespace ov::op::internal
