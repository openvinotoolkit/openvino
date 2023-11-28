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
    auto out_type = get_input_element_type(0);
    const auto input_size = get_input_size();
    for (size_t i = 1; i < input_size; i++) {
        if (i == 3 && (input_size == 4 || get_input_element_type(i) == element::boolean)) {
            continue;
        }
        OPENVINO_ASSERT(element::Type::merge(out_type, out_type, get_input_element_type(i)),
                        "Mixed input types are not supported.");
    }
    OPENVINO_ASSERT(out_type.is_real() || out_type.is_dynamic(),
                    "The element type of the input tensor must be a floating-point but got: ",
                    out_type);
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
