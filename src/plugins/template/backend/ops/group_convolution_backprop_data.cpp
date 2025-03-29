// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/group_convolution_backprop_data.hpp"

#include "evaluate_node.hpp"

template <ov::element::Type_t T>
bool evaluate(const std::shared_ptr<ov::op::v1::GroupConvolutionBackpropData>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    using ET = typename ov::element_type_traits<T>::value_type;
    const auto in_data_ptr = inputs[0].data<ET>();
    const auto filter_data_ptr = inputs[1].data<ET>();
    const auto out_data_ptr = outputs[0].data<ET>();
    const auto in_shape = inputs[0].get_shape();
    const auto filter_shape = inputs[1].get_shape();
    const auto out_shape = outputs[0].get_shape();
    ov::reference::group_convolution_backprop_data<ET>(in_data_ptr,
                                                       filter_data_ptr,
                                                       out_data_ptr,
                                                       in_shape,
                                                       filter_shape,
                                                       out_shape,
                                                       op->get_strides(),
                                                       op->get_dilations(),
                                                       op->get_pads_begin(),
                                                       op->get_pads_end(),
                                                       op->get_output_padding());
    return true;
}

template <>
bool evaluate_node<ov::op::v1::GroupConvolutionBackpropData>(std::shared_ptr<ov::Node> node,
                                                             ov::TensorVector& outputs,
                                                             const ov::TensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ov::op::v1::Select>(node) || ov::is_type<ov::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ov::element::boolean:
        return evaluate<ov::element::boolean>(ov::as_type_ptr<ov::op::v1::GroupConvolutionBackpropData>(node),
                                              outputs,
                                              inputs);
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(ov::as_type_ptr<ov::op::v1::GroupConvolutionBackpropData>(node),
                                           outputs,
                                           inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(ov::as_type_ptr<ov::op::v1::GroupConvolutionBackpropData>(node),
                                          outputs,
                                          inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64>(ov::as_type_ptr<ov::op::v1::GroupConvolutionBackpropData>(node),
                                          outputs,
                                          inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::v1::GroupConvolutionBackpropData>(node),
                                          outputs,
                                          inputs);
    case ov::element::i4:
        return evaluate<ov::element::i4>(ov::as_type_ptr<ov::op::v1::GroupConvolutionBackpropData>(node),
                                         outputs,
                                         inputs);
    case ov::element::i8:
        return evaluate<ov::element::i8>(ov::as_type_ptr<ov::op::v1::GroupConvolutionBackpropData>(node),
                                         outputs,
                                         inputs);
    case ov::element::i16:
        return evaluate<ov::element::i16>(ov::as_type_ptr<ov::op::v1::GroupConvolutionBackpropData>(node),
                                          outputs,
                                          inputs);
    case ov::element::i32:
        return evaluate<ov::element::i32>(ov::as_type_ptr<ov::op::v1::GroupConvolutionBackpropData>(node),
                                          outputs,
                                          inputs);
    case ov::element::i64:
        return evaluate<ov::element::i64>(ov::as_type_ptr<ov::op::v1::GroupConvolutionBackpropData>(node),
                                          outputs,
                                          inputs);
    case ov::element::u1:
        return evaluate<ov::element::u1>(ov::as_type_ptr<ov::op::v1::GroupConvolutionBackpropData>(node),
                                         outputs,
                                         inputs);
    case ov::element::u4:
        return evaluate<ov::element::u4>(ov::as_type_ptr<ov::op::v1::GroupConvolutionBackpropData>(node),
                                         outputs,
                                         inputs);
    case ov::element::u8:
        return evaluate<ov::element::u8>(ov::as_type_ptr<ov::op::v1::GroupConvolutionBackpropData>(node),
                                         outputs,
                                         inputs);
    case ov::element::u16:
        return evaluate<ov::element::u16>(ov::as_type_ptr<ov::op::v1::GroupConvolutionBackpropData>(node),
                                          outputs,
                                          inputs);
    case ov::element::u32:
        return evaluate<ov::element::u32>(ov::as_type_ptr<ov::op::v1::GroupConvolutionBackpropData>(node),
                                          outputs,
                                          inputs);
    case ov::element::u64:
        return evaluate<ov::element::u64>(ov::as_type_ptr<ov::op::v1::GroupConvolutionBackpropData>(node),
                                          outputs,
                                          inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", node->get_element_type().get_type_name(), " in evaluate_node()");
    }
}
