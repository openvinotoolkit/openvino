// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/scaled_dot_product_attention.hpp"

#include "itt.hpp"
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
                                                              bool causal)
    : ScaledDotProductAttention({query, key, value, attn_mask}, causal) {}

op::v13::ScaledDotProductAttention::ScaledDotProductAttention(const Output<Node>& query,
                                                              const Output<Node>& key,
                                                              const Output<Node>& value,
                                                              bool causal)
    : ScaledDotProductAttention({query, key, value}, causal) {}

void op::v13::ScaledDotProductAttention::validate_and_infer_types() {
    OV_OP_SCOPE(v13_ScaledDotProductAttention_validate_and_infer_types);

    for (size_t i = 0; i < get_input_size(); i++) {
        // TODO bool allowed for inp idx 4 Attn mask
        // TODO merge type for floats? Check transform
        OPENVINO_ASSERT(get_input_element_type(i).is_real() || get_input_element_type(i).is_dynamic(),
                        "The element type of the input tensor on index ",
                        i,
                        " must be a floating-point or dynamic (got ",
                        get_input_element_type(i),
                        ").");
    }
    OPENVINO_SUPPRESS_DEPRECATED_START
    const auto input_shapes = get_node_input_partial_shapes(*this);
    OPENVINO_SUPPRESS_DEPRECATED_END
    const auto output_shapes = shape_infer(this, input_shapes);
    set_output_type(0, get_input_element_type(0), output_shapes[0]);
}

std::shared_ptr<Node> op::v13::ScaledDotProductAttention::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v13_ScaledDotProductAttention_clone_with_new_inputs);
    return std::make_shared<ScaledDotProductAttention>(new_args, m_causal);
}

bool op::v13::ScaledDotProductAttention::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v13_ScaledDotProductAttention_visit_attributes);
    visitor.on_attribute("causal", m_causal);
    return true;
}

}  // namespace ov
