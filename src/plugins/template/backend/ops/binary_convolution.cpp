// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/binary_convolution.hpp"

#include "evaluate_node.hpp"

namespace bin_conv_v1 {
template <ngraph::element::Type_t t_in, ngraph::element::Type_t t_f>
inline void evaluate(const std::shared_ptr<ngraph::op::v1::BinaryConvolution>& op,
                     const ngraph::HostTensorVector& outputs,
                     const ngraph::HostTensorVector& inputs) {
    using T_IN = typename ngraph::element_type_traits<t_in>::value_type;
    using T_F = typename ngraph::element_type_traits<t_f>::value_type;

    const auto in_data_ptr = inputs[0]->get_data_ptr<T_IN>();
    const auto filter_data_ptr = inputs[1]->get_data_ptr<T_F>();
    auto out_data_ptr = outputs[0]->get_data_ptr<T_IN>();
    const auto in_shape = inputs[0]->get_shape();
    const auto filter_shape = inputs[1]->get_shape();
    const auto out_shape = outputs[0]->get_shape();

    ov::reference::binary_convolution<T_IN, T_F>(in_data_ptr,
                                                 filter_data_ptr,
                                                 out_data_ptr,
                                                 in_shape,
                                                 filter_shape,
                                                 out_shape,
                                                 op->get_strides(),
                                                 op->get_dilations(),
                                                 op->get_pads_begin(),
                                                 op->get_pads_end(),
                                                 op->get_pad_value());
}
}  // namespace bin_conv_v1

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v1::BinaryConvolution>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    switch (inputs[1]->get_element_type()) {
    case ngraph::element::Type_t::u1:
        bin_conv_v1::evaluate<ET, ngraph::element::Type_t::u8>(op, outputs, inputs);
        break;
    default:
        throw std::runtime_error("BinaryConvolution supports only u1 element type for filters input");
        break;
    }
    return true;
}

template <>
bool evaluate_node<ngraph::op::v1::BinaryConvolution>(std::shared_ptr<ngraph::Node> node,
                                                      const ngraph::HostTensorVector& outputs,
                                                      const ngraph::HostTensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ngraph::op::v1::Select>(node) || ov::is_type<ngraph::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ngraph::element::Type_t::boolean:
        return evaluate<ngraph::element::Type_t::boolean>(ov::as_type_ptr<ngraph::op::v1::BinaryConvolution>(node),
                                                          outputs,
                                                          inputs);
    case ngraph::element::Type_t::bf16:
        return evaluate<ngraph::element::Type_t::bf16>(ov::as_type_ptr<ngraph::op::v1::BinaryConvolution>(node),
                                                       outputs,
                                                       inputs);
    case ngraph::element::Type_t::f16:
        return evaluate<ngraph::element::Type_t::f16>(ov::as_type_ptr<ngraph::op::v1::BinaryConvolution>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::f64:
        return evaluate<ngraph::element::Type_t::f64>(ov::as_type_ptr<ngraph::op::v1::BinaryConvolution>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::f32:
        return evaluate<ngraph::element::Type_t::f32>(ov::as_type_ptr<ngraph::op::v1::BinaryConvolution>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::i4:
        return evaluate<ngraph::element::Type_t::i4>(ov::as_type_ptr<ngraph::op::v1::BinaryConvolution>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::i8:
        return evaluate<ngraph::element::Type_t::i8>(ov::as_type_ptr<ngraph::op::v1::BinaryConvolution>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::i16:
        return evaluate<ngraph::element::Type_t::i16>(ov::as_type_ptr<ngraph::op::v1::BinaryConvolution>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::i32:
        return evaluate<ngraph::element::Type_t::i32>(ov::as_type_ptr<ngraph::op::v1::BinaryConvolution>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::i64:
        return evaluate<ngraph::element::Type_t::i64>(ov::as_type_ptr<ngraph::op::v1::BinaryConvolution>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::u1:
        return evaluate<ngraph::element::Type_t::u1>(ov::as_type_ptr<ngraph::op::v1::BinaryConvolution>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::u4:
        return evaluate<ngraph::element::Type_t::u4>(ov::as_type_ptr<ngraph::op::v1::BinaryConvolution>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::u8:
        return evaluate<ngraph::element::Type_t::u8>(ov::as_type_ptr<ngraph::op::v1::BinaryConvolution>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::u16:
        return evaluate<ngraph::element::Type_t::u16>(ov::as_type_ptr<ngraph::op::v1::BinaryConvolution>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::u32:
        return evaluate<ngraph::element::Type_t::u32>(ov::as_type_ptr<ngraph::op::v1::BinaryConvolution>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::u64:
        return evaluate<ngraph::element::Type_t::u64>(ov::as_type_ptr<ngraph::op::v1::BinaryConvolution>(node),
                                                      outputs,
                                                      inputs);
    default:
        OPENVINO_THROW(std::string("Unhandled data type ") + node->get_element_type().get_type_name() +
                       std::string("in evaluate_node()"));
    }
}
