// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/scaled_dot_product_attention.hpp"

#include "evaluate_node.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "scaled_dot_product_attention_shape_inference.hpp"

template <ov::element::Type_t ET, ov::element::Type_t ETMask>
bool evaluate(const std::shared_ptr<ov::op::v13::ScaledDotProductAttention>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    using T = typename ov::element_type_traits<ET>::value_type;
    using TMask = typename ov::element_type_traits<ETMask>::value_type;
    const TMask* mask = inputs.size() == 4 ? inputs[3].data<const TMask>() : nullptr;
    const T* scale = inputs.size() == 5 ? inputs[4].data<const T>() : nullptr;
    auto mask_shape = inputs.size() == 4 ? inputs[3].get_shape() : ov::Shape{};
    const T* sink = inputs.size() == 6 ? inputs[5].data<const T>() : nullptr;
    auto sink_shape = inputs.size() == 6 ? inputs[5].get_shape() : ov::Shape{};

    // Hack below is needed to support dynamic shapes in the reference implementation...
    const auto input_shapes = ov::util::get_tensors_partial_shapes(inputs);
    const auto output_shape =
        ov::op::v13::shape_infer(op.get(), input_shapes, make_tensor_accessor(inputs)).front().to_shape();
    outputs[0].set_shape(output_shape);
    // ---

    ov::reference::scaled_dot_product_attention<T, TMask>(inputs[0].data<const T>(),
                                                          inputs[1].data<const T>(),
                                                          inputs[2].data<const T>(),
                                                          mask,
                                                          scale,
                                                          sink,
                                                          outputs[0].data<T>(),
                                                          op->get_causal(),
                                                          inputs[0].get_shape(),
                                                          inputs[1].get_shape(),
                                                          inputs[2].get_shape(),
                                                          mask_shape,
                                                          sink_shape,
                                                          outputs[0].get_shape());

    return true;
}

template <>
bool evaluate_node<ov::op::v13::ScaledDotProductAttention>(std::shared_ptr<ov::Node> node,
                                                           ov::TensorVector& outputs,
                                                           const ov::TensorVector& inputs) {
    const auto& element_type = node->get_input_element_type(0);
    const auto& mask_element_type = node->get_input_size() >= 4 ? node->get_input_element_type(3) : element_type;
#define CASE(type)                                                             \
    case ov::element::type: {                                                  \
        if (mask_element_type == ov::element::boolean) {                       \
            return evaluate<ov::element::type, ov::element::boolean>(          \
                ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(node), \
                outputs,                                                       \
                inputs);                                                       \
        } else {                                                               \
            return evaluate<ov::element::type, ov::element::type>(             \
                ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(node), \
                outputs,                                                       \
                inputs);                                                       \
        }                                                                      \
    }

    switch (element_type) {
        CASE(bf16);
        CASE(f16);
        CASE(f32);
        CASE(f64);
    default:
        OPENVINO_THROW("Unhandled data type ", element_type, " in evaluate_node()");
    }

#undef CASE
}