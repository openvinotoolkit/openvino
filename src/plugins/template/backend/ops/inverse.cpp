// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/inverse.hpp"

#include "evaluate_node.hpp"
#include "inverse_shape_inference.hpp"
#include "openvino/reference/convert.hpp"

template <ov::element::Type_t ET>
inline bool evaluate(const std::shared_ptr<ov::op::v14::Inverse>& op,
                     ov::TensorVector& outputs,
                     const ov::TensorVector& inputs) {
    using T = typename ov::element_type_traits<ET>::value_type;

    const std::vector<ov::PartialShape> input_shapes{op->get_input_shape(0)};
    const auto out_shape = ov::op::v14::shape_infer(op.get(), input_shapes).front().to_shape();
    outputs[0].set_shape(out_shape);

    ov::reference::inverse<T>(inputs[0].data<const T>(), outputs[0].data<T>(), out_shape, op->get_adjoint());
    return true;
}

template <ov::element::Type_t ET>
inline bool evaluate_bf16(const std::shared_ptr<ov::op::v14::Inverse>& op,
                          ov::TensorVector& outputs,
                          const ov::TensorVector& inputs) {
    using T = typename ov::element_type_traits<ET>::value_type;

    const std::vector<ov::PartialShape> input_shapes{op->get_input_shape(0)};
    const auto out_shape = ov::op::v14::shape_infer(op.get(), input_shapes).front().to_shape();
    outputs[0].set_shape(out_shape);

    size_t element_count = ov::shape_size(outputs[0].get_shape());
    ov::TensorVector convert_to_float_in{ov::Tensor(ov::element::f32, out_shape)};
    ov::TensorVector convert_to_float_out{ov::Tensor(ov::element::f32, out_shape)};

    ov::reference::convert<T, float>(inputs[0].data<const T>(), convert_to_float_in[0].data<float>(), element_count);
    ov::reference::inverse<float>(convert_to_float_in[0].data<float>(),
                                  convert_to_float_out[0].data<float>(),
                                  out_shape,
                                  op->get_adjoint());
    ov::reference::convert<float, T>(convert_to_float_out[0].data<float>(), outputs[0].data<T>(), element_count);

    return true;
}

template <>
bool evaluate_node<ov::op::v14::Inverse>(std::shared_ptr<ov::Node> node,
                                         ov::TensorVector& outputs,
                                         const ov::TensorVector& inputs) {
    switch (node->get_input_element_type(0)) {
    case ov::element::Type_t::f16:
        return evaluate<ov::element::Type_t::f16>(ov::as_type_ptr<ov::op::v14::Inverse>(node), outputs, inputs);
    case ov::element::Type_t::f32:
        return evaluate<ov::element::Type_t::f32>(ov::as_type_ptr<ov::op::v14::Inverse>(node), outputs, inputs);
    case ov::element::Type_t::f64:
        return evaluate<ov::element::Type_t::f64>(ov::as_type_ptr<ov::op::v14::Inverse>(node), outputs, inputs);
    case ov::element::Type_t::bf16:
        return evaluate_bf16<ov::element::Type_t::bf16>(ov::as_type_ptr<ov::op::v14::Inverse>(node), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled input data type ",
                       node->get_input_element_type(0).get_type_name(),
                       " in evaluate_node().");
    }
}
