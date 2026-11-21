// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/scaled_dot_product_attention.hpp"

#include "itt.hpp"
#include "openvino/op/util/sdpa_util.hpp"
#include "scaled_dot_product_attention_shape_inference.hpp"

using namespace std;
namespace ov {

op::v13::ScaledDotProductAttention::ScaledDotProductAttention(const OutputVector& inputs, bool causal)
    : op::Op(inputs),
      m_causal(causal) {
    constructor_validate_and_infer_types();
}

op::v13::ScaledDotProductAttention::ScaledDotProductAttention(const Output<Node>& query,
                                                              const Output<Node>& key,
                                                              const Output<Node>& value,
                                                              const Output<Node>& attn_mask,
                                                              const Output<Node>& scale,
                                                              bool causal)
    : ScaledDotProductAttention({query, key, value, attn_mask, scale}, causal) {}

op::v13::ScaledDotProductAttention::ScaledDotProductAttention(const Output<Node>& query,
                                                              const Output<Node>& key,
                                                              const Output<Node>& value,
                                                              const Output<Node>& attn_mask,
                                                              const Output<Node>& scale,
                                                              const Output<Node>& sink,
                                                              bool causal)
    : ScaledDotProductAttention({query, key, value, attn_mask, scale, sink}, causal) {}

op::v13::ScaledDotProductAttention::ScaledDotProductAttention(const Output<Node>& query,
                                                              const Output<Node>& key,
                                                              const Output<Node>& value,
                                                              const Output<Node>& attn_mask,
                                                              bool causal)
    : ScaledDotProductAttention({query, key, value, attn_mask}, causal) {}

op::v13::ScaledDotProductAttention::ScaledDotProductAttention(const Output<Node>& query,
                                                              const Output<Node>& key,
                                                              const Output<Node>& value,
                                                              bool causal)
    : ScaledDotProductAttention({query, key, value}, causal) {}

void op::v13::ScaledDotProductAttention::validate_and_infer_types() {
    OV_OP_SCOPE(v13_ScaledDotProductAttention_validate_and_infer_types);
    auto out_type = get_input_element_type(0);
    const auto& input_size = get_input_size();
    const auto& causal = get_causal();
    if (input_size >= 4 && !causal) {
        const auto& attention_type = get_input_element_type(3);
        NODE_VALIDATION_CHECK(
            this,
            attention_type.is_real() || attention_type == element::boolean || attention_type.is_dynamic(),
            "The element type of attention_mask must be either floating-point or boolean.");
    }
    const auto merge_input_type = [&](size_t i) {
        NODE_VALIDATION_CHECK(this,
                              element::Type::merge(out_type, out_type, get_input_element_type(i)),
                              "Mixed input types are not supported.");
    };
    // key and value: T, or T_QUANT, which the operation converts to T.
    for (size_t i = 1; i <= 2; i++) {
        if (!util::is_sdpa_integer_kv_type(get_input_element_type(i))) {
            merge_input_type(i);
        }
    }
    // attention_mask: T, or boolean; ignored when causal.
    if (input_size >= 4 && !causal && get_input_element_type(3) != element::boolean) {
        merge_input_type(3);
    }
    // scale and sink: T.
    for (size_t i = 4; i < input_size; i++) {
        merge_input_type(i);
    }
    NODE_VALIDATION_CHECK(this,
                          out_type.is_real() || out_type.is_dynamic(),
                          "The element type of the input tensor must be a floating-point.");

    const auto& input_shapes = ov::util::get_node_input_partial_shapes(*this);
    const auto output_shapes = shape_infer(this, input_shapes);
    set_output_type(0, out_type, output_shapes[0]);
}

std::shared_ptr<Node> op::v13::ScaledDotProductAttention::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v13_ScaledDotProductAttention_clone_with_new_inputs);
    return std::make_shared<ScaledDotProductAttention>(new_args, m_causal);
}

bool op::util::is_sdpa_integer_kv_type(const element::Type& type) {
    // The signed and unsigned 8- and 4-bit integer types, which is what a quantized key or value
    // cache is stored as.
    return type == element::i8 || type == element::u8 || type == element::i4 || type == element::u4;
}

bool op::util::sdpa_has_integer_kv(const op::v13::ScaledDotProductAttention& sdpa) {
    return is_sdpa_integer_kv_type(sdpa.get_input_element_type(1)) ||
           is_sdpa_integer_kv_type(sdpa.get_input_element_type(2));
}

bool op::v13::ScaledDotProductAttention::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v13_ScaledDotProductAttention_visit_attributes);
    visitor.on_attribute("causal", m_causal);
    return true;
}

}  // namespace ov
