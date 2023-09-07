// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/roll.hpp"

#include "evaluate_node.hpp"

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v7::Roll>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    const auto& shiftType = inputs[1]->get_element_type();
    std::vector<int64_t> shift_int64;
    if (shiftType == ngraph::element::Type_t::i32) {
        auto shift = inputs[1]->get_data_ptr<const int32_t>();
        shift_int64.resize(ngraph::shape_size(inputs[1]->get_shape()));
        std::transform(shift,
                       shift + ngraph::shape_size(inputs[1]->get_shape()),
                       shift_int64.begin(),
                       [](const int32_t& elem) {
                           return static_cast<int64_t>(elem);
                       });
    }
    const auto& axesType = inputs[2]->get_element_type();
    std::vector<int64_t> axes_int64;
    if (axesType == ngraph::element::Type_t::i32) {
        auto axes = inputs[2]->get_data_ptr<const int32_t>();
        axes_int64.resize(ngraph::shape_size(inputs[2]->get_shape()));
        std::transform(axes,
                       axes + ngraph::shape_size(inputs[2]->get_shape()),
                       axes_int64.begin(),
                       [](const int32_t& elem) {
                           return static_cast<int64_t>(elem);
                       });
    }
    ov::reference::roll(
        inputs[0]->get_data_ptr<const char>(),
        inputs[1]->get_element_type() != ngraph::element::Type_t::i64 ? shift_int64.data()
                                                                      : inputs[1]->get_data_ptr<const int64_t>(),
        inputs[2]->get_element_type() != ngraph::element::Type_t::i64 ? axes_int64.data()
                                                                      : inputs[2]->get_data_ptr<const int64_t>(),
        outputs[0]->get_data_ptr<char>(),
        inputs[0]->get_shape(),
        inputs[1]->get_shape(),
        inputs[2]->get_shape(),
        inputs[0]->get_element_type().size());
    return true;
}

template <>
bool evaluate_node<ngraph::op::v7::Roll>(std::shared_ptr<ngraph::Node> node,
                                         const ngraph::HostTensorVector& outputs,
                                         const ngraph::HostTensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ngraph::op::v1::Select>(node) || ov::is_type<ngraph::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ngraph::element::Type_t::boolean:
        return evaluate<ngraph::element::Type_t::boolean>(ov::as_type_ptr<ngraph::op::v7::Roll>(node), outputs, inputs);
    case ngraph::element::Type_t::bf16:
        return evaluate<ngraph::element::Type_t::bf16>(ov::as_type_ptr<ngraph::op::v7::Roll>(node), outputs, inputs);
    case ngraph::element::Type_t::f16:
        return evaluate<ngraph::element::Type_t::f16>(ov::as_type_ptr<ngraph::op::v7::Roll>(node), outputs, inputs);
    case ngraph::element::Type_t::f64:
        return evaluate<ngraph::element::Type_t::f64>(ov::as_type_ptr<ngraph::op::v7::Roll>(node), outputs, inputs);
    case ngraph::element::Type_t::f32:
        return evaluate<ngraph::element::Type_t::f32>(ov::as_type_ptr<ngraph::op::v7::Roll>(node), outputs, inputs);
    case ngraph::element::Type_t::i4:
        return evaluate<ngraph::element::Type_t::i4>(ov::as_type_ptr<ngraph::op::v7::Roll>(node), outputs, inputs);
    case ngraph::element::Type_t::i8:
        return evaluate<ngraph::element::Type_t::i8>(ov::as_type_ptr<ngraph::op::v7::Roll>(node), outputs, inputs);
    case ngraph::element::Type_t::i16:
        return evaluate<ngraph::element::Type_t::i16>(ov::as_type_ptr<ngraph::op::v7::Roll>(node), outputs, inputs);
    case ngraph::element::Type_t::i32:
        return evaluate<ngraph::element::Type_t::i32>(ov::as_type_ptr<ngraph::op::v7::Roll>(node), outputs, inputs);
    case ngraph::element::Type_t::i64:
        return evaluate<ngraph::element::Type_t::i64>(ov::as_type_ptr<ngraph::op::v7::Roll>(node), outputs, inputs);
    case ngraph::element::Type_t::u1:
        return evaluate<ngraph::element::Type_t::u1>(ov::as_type_ptr<ngraph::op::v7::Roll>(node), outputs, inputs);
    case ngraph::element::Type_t::u4:
        return evaluate<ngraph::element::Type_t::u4>(ov::as_type_ptr<ngraph::op::v7::Roll>(node), outputs, inputs);
    case ngraph::element::Type_t::u8:
        return evaluate<ngraph::element::Type_t::u8>(ov::as_type_ptr<ngraph::op::v7::Roll>(node), outputs, inputs);
    case ngraph::element::Type_t::u16:
        return evaluate<ngraph::element::Type_t::u16>(ov::as_type_ptr<ngraph::op::v7::Roll>(node), outputs, inputs);
    case ngraph::element::Type_t::u32:
        return evaluate<ngraph::element::Type_t::u32>(ov::as_type_ptr<ngraph::op::v7::Roll>(node), outputs, inputs);
    case ngraph::element::Type_t::u64:
        return evaluate<ngraph::element::Type_t::u64>(ov::as_type_ptr<ngraph::op::v7::Roll>(node), outputs, inputs);
    default:
        OPENVINO_THROW(std::string("Unhandled data type ") + node->get_element_type().get_type_name() +
                       std::string("in evaluate_node()"));
    }
}
