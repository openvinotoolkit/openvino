// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/scaled_dot_product_attention_extension.hpp"

#include "openvino/core/validation_util.hpp"
#include "openvino/reference/scaled_dot_product_attention.hpp"
#include "scaled_dot_product_attention_shape_inference.hpp"

namespace ov::op::internal {

ScaledDotProductAttentionExtension::ScaledDotProductAttentionExtension(const OutputVector& inputs,
                                                                       bool causal,
                                                                       element::Type query_precision,
                                                                       element::Type probability_precision)
    : v13::ScaledDotProductAttention(inputs, causal),
      m_query_precision(query_precision),
      m_probability_precision(probability_precision) {
    constructor_validate_and_infer_types();
}

void ScaledDotProductAttentionExtension::validate_and_infer_types() {
    for (const auto precision : {m_query_precision, m_probability_precision}) {
        NODE_VALIDATION_CHECK(this,
                              precision == element::dynamic || precision == element::f16,
                              "Attention rounding precision must be dynamic or f16");
    }
    v13::ScaledDotProductAttention::validate_and_infer_types();
    if (m_query_precision != element::dynamic || m_probability_precision != element::dynamic) {
        NODE_VALIDATION_CHECK(this,
                              get_input_element_type(0) == element::f32 || get_input_element_type(0) == element::f16,
                              "Explicit attention rounding requires f32 or f16 inputs");
        set_output_type(0, element::f32, get_output_partial_shape(0));
    }
}

std::shared_ptr<Node> ScaledDotProductAttentionExtension::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<ScaledDotProductAttentionExtension>(new_args,
                                                                get_causal(),
                                                                m_query_precision,
                                                                m_probability_precision);
}

bool ScaledDotProductAttentionExtension::visit_attributes(AttributeVisitor& visitor) {
    v13::ScaledDotProductAttention::visit_attributes(visitor);
    visitor.on_attribute("query_precision", m_query_precision);
    visitor.on_attribute("probability_precision", m_probability_precision);
    return true;
}

bool ScaledDotProductAttentionExtension::has_evaluate() const {
    return get_input_element_type(0) == element::f32;
}

bool ScaledDotProductAttentionExtension::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    if (!has_evaluate())
        return false;
    const auto input_shapes = ov::util::get_tensors_partial_shapes(inputs);
    outputs[0].set_shape(v13::shape_infer(this, input_shapes, make_tensor_accessor(inputs)).front().to_shape());
    const auto run = [&](auto mask) {
        ov::reference::scaled_dot_product_attention<float>(inputs[0].data<const float>(),
                                                           inputs[1].data<const float>(),
                                                           inputs[2].data<const float>(),
                                                           mask,
                                                           inputs.size() > 4 ? inputs[4].data<const float>() : nullptr,
                                                           inputs.size() > 5 ? inputs[5].data<const float>() : nullptr,
                                                           outputs[0].data<float>(),
                                                           get_causal(),
                                                           inputs[0].get_shape(),
                                                           inputs[1].get_shape(),
                                                           inputs[2].get_shape(),
                                                           inputs.size() > 3 ? inputs[3].get_shape() : Shape{},
                                                           inputs.size() > 5 ? inputs[5].get_shape() : Shape{},
                                                           outputs[0].get_shape(),
                                                           m_query_precision == element::f16,
                                                           m_probability_precision == element::f16);
    };
    if (get_causal())
        run(static_cast<const float*>(nullptr));
    else if (inputs.size() > 3 && inputs[3].get_element_type() == element::boolean)
        run(inputs[3].data<const char>());
    else
        run(inputs.size() > 3 ? inputs[3].data<const float>() : nullptr);
    return true;
}

}  // namespace ov::op::internal
