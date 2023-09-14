// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/roll.hpp"

#include "evaluate_node.hpp"

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v7::Roll>& op, ov::TensorVector& outputs, const ov::TensorVector& inputs) {
    const auto& shiftType = inputs[1].get_element_type();
    std::vector<int64_t> shift_int64;
    if (shiftType == ov::element::i32) {
        auto shift = inputs[1].data<const int32_t>();
        shift_int64.resize(ov::shape_size(inputs[1].get_shape()));
        std::transform(shift,
                       shift + ov::shape_size(inputs[1].get_shape()),
                       shift_int64.begin(),
                       [](const int32_t& elem) {
                           return static_cast<int64_t>(elem);
                       });
    }
    const auto& axesType = inputs[2].get_element_type();
    std::vector<int64_t> axes_int64;
    if (axesType == ov::element::i32) {
        auto axes = inputs[2].data<const int32_t>();
        axes_int64.resize(ov::shape_size(inputs[2].get_shape()));
        std::transform(axes, axes + ov::shape_size(inputs[2].get_shape()), axes_int64.begin(), [](const int32_t& elem) {
            return static_cast<int64_t>(elem);
        });
    }
    ov::reference::roll(
        inputs[0].data<const char>(),
        inputs[1].get_element_type() != ov::element::i64 ? shift_int64.data() : inputs[1].data<const int64_t>(),
        inputs[2].get_element_type() != ov::element::i64 ? axes_int64.data() : inputs[2].data<const int64_t>(),
        outputs[0].data<char>(),
        inputs[0].get_shape(),
        inputs[1].get_shape(),
        inputs[2].get_shape(),
        inputs[0].get_element_type().size());
    return true;
}

template <>
bool evaluate_node<ov::op::v7::Roll>(std::shared_ptr<ov::Node> node,
                                     ov::TensorVector& outputs,
                                     const ov::TensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ov::op::v1::Select>(node) || ov::is_type<ov::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ov::element::boolean:
        return evaluate<ov::element::boolean>(ov::as_type_ptr<ov::op::v7::Roll>(node), outputs, inputs);
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(ov::as_type_ptr<ov::op::v7::Roll>(node), outputs, inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(ov::as_type_ptr<ov::op::v7::Roll>(node), outputs, inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64>(ov::as_type_ptr<ov::op::v7::Roll>(node), outputs, inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::v7::Roll>(node), outputs, inputs);
    case ov::element::i4:
        return evaluate<ov::element::i4>(ov::as_type_ptr<ov::op::v7::Roll>(node), outputs, inputs);
    case ov::element::i8:
        return evaluate<ov::element::i8>(ov::as_type_ptr<ov::op::v7::Roll>(node), outputs, inputs);
    case ov::element::i16:
        return evaluate<ov::element::i16>(ov::as_type_ptr<ov::op::v7::Roll>(node), outputs, inputs);
    case ov::element::i32:
        return evaluate<ov::element::i32>(ov::as_type_ptr<ov::op::v7::Roll>(node), outputs, inputs);
    case ov::element::i64:
        return evaluate<ov::element::i64>(ov::as_type_ptr<ov::op::v7::Roll>(node), outputs, inputs);
    case ov::element::u1:
        return evaluate<ov::element::u1>(ov::as_type_ptr<ov::op::v7::Roll>(node), outputs, inputs);
    case ov::element::u4:
        return evaluate<ov::element::u4>(ov::as_type_ptr<ov::op::v7::Roll>(node), outputs, inputs);
    case ov::element::u8:
        return evaluate<ov::element::u8>(ov::as_type_ptr<ov::op::v7::Roll>(node), outputs, inputs);
    case ov::element::u16:
        return evaluate<ov::element::u16>(ov::as_type_ptr<ov::op::v7::Roll>(node), outputs, inputs);
    case ov::element::u32:
        return evaluate<ov::element::u32>(ov::as_type_ptr<ov::op::v7::Roll>(node), outputs, inputs);
    case ov::element::u64:
        return evaluate<ov::element::u64>(ov::as_type_ptr<ov::op::v7::Roll>(node), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", node->get_element_type().get_type_name(), " in evaluate_node()");
    }
}
