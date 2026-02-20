// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/convolution.hpp"

#include "convolution_shape_inference.hpp"
#include "evaluate_node.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/op/convolution.hpp"

template <ov::element::Type_t T>
bool evaluate(const std::shared_ptr<ov::op::v1::Convolution>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    using ET = typename ov::element_type_traits<T>::value_type;
    const auto filter_data = inputs[1].data<ET>();
    const auto in_data_ptr = inputs[0].data<ET>();
    const auto& in_shape = inputs[0].get_shape();
    const auto& filter_shape = inputs[1].get_shape();

    const auto input_shapes = ov::util::get_tensors_partial_shapes(inputs);
    auto bpads_cp = ov::CoordinateDiff();
    auto endpads_cp = ov::CoordinateDiff();
    const auto output_shape = ov::op::v1::shape_infer(op.get(), input_shapes, bpads_cp, endpads_cp).front();
    ov::Shape out_shape = output_shape.to_shape();
    outputs[0].set_shape(out_shape);
    auto out_data_ptr = outputs[0].data<ET>();

    ov::reference::convolution<ET>(in_data_ptr,
                                   filter_data,
                                   out_data_ptr,
                                   in_shape,
                                   filter_shape,
                                   out_shape,
                                   op->get_strides(),
                                   op->get_dilations(),
                                   bpads_cp,
                                   endpads_cp);
    return true;
}

template <>
bool evaluate_node<ov::op::v1::Convolution>(std::shared_ptr<ov::Node> node,
                                            ov::TensorVector& outputs,
                                            const ov::TensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ov::op::v1::Select>(node) || ov::is_type<ov::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ov::element::boolean:
        return evaluate<ov::element::boolean>(ov::as_type_ptr<ov::op::v1::Convolution>(node), outputs, inputs);
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(ov::as_type_ptr<ov::op::v1::Convolution>(node), outputs, inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(ov::as_type_ptr<ov::op::v1::Convolution>(node), outputs, inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64>(ov::as_type_ptr<ov::op::v1::Convolution>(node), outputs, inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::v1::Convolution>(node), outputs, inputs);
    case ov::element::i4:
        return evaluate<ov::element::i4>(ov::as_type_ptr<ov::op::v1::Convolution>(node), outputs, inputs);
    case ov::element::i8:
        return evaluate<ov::element::i8>(ov::as_type_ptr<ov::op::v1::Convolution>(node), outputs, inputs);
    case ov::element::i16:
        return evaluate<ov::element::i16>(ov::as_type_ptr<ov::op::v1::Convolution>(node), outputs, inputs);
    case ov::element::i32:
        return evaluate<ov::element::i32>(ov::as_type_ptr<ov::op::v1::Convolution>(node), outputs, inputs);
    case ov::element::i64:
        return evaluate<ov::element::i64>(ov::as_type_ptr<ov::op::v1::Convolution>(node), outputs, inputs);
    case ov::element::u1:
        return evaluate<ov::element::u1>(ov::as_type_ptr<ov::op::v1::Convolution>(node), outputs, inputs);
    case ov::element::u4:
        return evaluate<ov::element::u4>(ov::as_type_ptr<ov::op::v1::Convolution>(node), outputs, inputs);
    case ov::element::u8:
        return evaluate<ov::element::u8>(ov::as_type_ptr<ov::op::v1::Convolution>(node), outputs, inputs);
    case ov::element::u16:
        return evaluate<ov::element::u16>(ov::as_type_ptr<ov::op::v1::Convolution>(node), outputs, inputs);
    case ov::element::u32:
        return evaluate<ov::element::u32>(ov::as_type_ptr<ov::op::v1::Convolution>(node), outputs, inputs);
    case ov::element::u64:
        return evaluate<ov::element::u64>(ov::as_type_ptr<ov::op::v1::Convolution>(node), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", node->get_element_type().get_type_name(), " in evaluate_node()");
    }
}
