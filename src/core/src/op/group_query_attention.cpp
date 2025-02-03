// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/group_query_attention.hpp"

#include "itt.hpp"
#include "openvino/op/null.hpp"

using namespace std;
namespace ov {
namespace op {
namespace v15 {

GroupQueryAttention::GroupQueryAttention(const OutputVector& args,
                                         unsigned int num_heads,
                                         unsigned int kv_num_heads,
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

int64_t get_head_size(const PartialShape& input_shape, int num_heads, int kv_num_heads) {
    return input_shape[2].get_length() / (num_heads + kv_num_heads * 2);
}

std::vector<int64_t> get_qkv_sizes(const PartialShape& input_shape, int num_heads, int kv_num_heads) {
    int64_t per_head_size = get_head_size(input_shape, num_heads, kv_num_heads);
    const std::vector<int64_t> qkv_sizes = {num_heads * per_head_size,
                                            kv_num_heads * per_head_size,
                                            kv_num_heads * per_head_size};
    return qkv_sizes;
}

void GroupQueryAttention::validate_and_infer_types() {
    OV_OP_SCOPE(v15_GroupQueryAttention_validate_and_infer_types);
    PartialShape input_shape = get_input_partial_shape(0);
    Dimension batch_size = input_shape[0];
    Dimension sequence_len = input_shape[1];
    Dimension head_size;
    if (Null::is_null(input_value(1)) && Null::is_null(input_value(2))) {
        head_size = get_head_size(input_shape, m_num_heads, m_kv_num_heads);
    } else {
        head_size = input_shape[2].get_length() / m_num_heads;
    }
    Dimension output_kv_len;
    PartialShape kv_past_shape = get_input_partial_shape(3);
    // FIXME: https://github.com/openvinotoolkit/openvino/pull/27648
    if (kv_past_shape[2].is_static()) {
        output_kv_len = kv_past_shape[2] + sequence_len;
    } else {
        output_kv_len = ov::Dimension();
    }
    auto element_type = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          element_type == element::f32 || element_type == element::f16,
                          "GroupQueryAttention only suuports f32 and f16");
    set_output_type(0, element_type, PartialShape{batch_size, sequence_len, head_size * m_num_heads});
    set_output_type(1, element_type, PartialShape{batch_size, m_kv_num_heads, output_kv_len, head_size});
    set_output_type(2, element_type, PartialShape{batch_size, m_kv_num_heads, output_kv_len, head_size});
}

bool GroupQueryAttention::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v15_GroupQueryAttention_visit_attributes);
    visitor.on_attribute("do_rotary", m_do_rotary);
    visitor.on_attribute("kv_num_heads", m_kv_num_heads);
    visitor.on_attribute("num_heads", m_num_heads);
    visitor.on_attribute("rotary_interleaved", m_rotary_interleaved);
    visitor.on_attribute("scale", m_scale);
    return true;
}

std::shared_ptr<ov::Node> GroupQueryAttention::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OV_OP_SCOPE(v15_GroupQueryAttention_clone_with_new_inputs);
    return std::make_shared<GroupQueryAttention>(new_args,
                                                 m_num_heads,
                                                 m_kv_num_heads,
                                                 m_scale,
                                                 m_do_rotary,
                                                 m_rotary_interleaved);
}

}  // namespace v15
}  // namespace op
}  // namespace ov
