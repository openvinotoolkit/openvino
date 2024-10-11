// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/Identity.hpp"

#include "Identity_shape_inference.hpp"
#include "evaluate_node.hpp"

template <ov::element::Type_t ET>
inline bool evaluate(const std::shared_ptr<ov::op::v15::Identity>& op,
                     ov::TensorVector& outputs,
                     const ov::TensorVector& inputs) {
    using T = typename ov::element_type_traits<ET>::value_type;

    const std::vector<ov::PartialShape> input_shapes{op->get_input_shape(0)};
    const auto total_size = get_shape_size(out_shape);
    const auto total_size_in_bytes = total_size * inputs[0].get_dtype().get_element_size();

    outputs[0].set_shape(input_shapes[0]);

    ov::reference::Identity(static_cast<const char*>(inputs[0].data()), static_cast<char*>(outputs[0].data()), total_size_in_bytes);
    return true;
}

template <>
bool evaluate_node<ov::op::v15::Identity>(std::shared_ptr<ov::Node> node,
                                          ov::TensorVector& outputs,
                                          const ov::TensorVector& inputs) {
    switch (node->get_input_element_type(0)) {
    case ov::element::boolean:
        return evaluate<ov::element::boolean>(ov::as_type_ptr<ov::op::v15::Identity>(node), outputs, inputs);
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(ov::as_type_ptr<ov::op::v15::Identity>(node), outputs, inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(ov::as_type_ptr<ov::op::v15::Identity>(node), outputs, inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64>(ov::as_type_ptr<ov::op::v15::Identity>(node), outputs, inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::v15::Identity>(node), outputs, inputs);
    case ov::element::i4:
        return evaluate<ov::element::i4>(ov::as_type_ptr<ov::op::v15::Identity>(node), outputs, inputs);
    case ov::element::i8:
        return evaluate<ov::element::i8>(ov::as_type_ptr<ov::op::v15::Identity>(node), outputs, inputs);
    case ov::element::i16:
        return evaluate<ov::element::i16>(ov::as_type_ptr<ov::op::v15::Identity>(node), outputs, inputs);
    case ov::element::i32:
        return evaluate<ov::element::i32>(ov::as_type_ptr<ov::op::v15::Identity>(node), outputs, inputs);
    case ov::element::i64:
        return evaluate<ov::element::i64>(ov::as_type_ptr<ov::op::v15::Identity>(node), outputs, inputs);
    case ov::element::u1:
        return evaluate<ov::element::u1>(ov::as_type_ptr<ov::op::v15::Identity>(node), outputs, inputs);
    case ov::element::u4:
        return evaluate<ov::element::u4>(ov::as_type_ptr<ov::op::v15::Identity>(node), outputs, inputs);
    case ov::element::u8:
        return evaluate<ov::element::u8>(ov::as_type_ptr<ov::op::v15::Identity>(node), outputs, inputs);
    case ov::element::u16:
        return evaluate<ov::element::u16>(ov::as_type_ptr<ov::op::v15::Identity>(node), outputs, inputs);
    case ov::element::u32:
        return evaluate<ov::element::u32>(ov::as_type_ptr<ov::op::v15::Identity>(node), outputs, inputs);
    case ov::element::u64:
        return evaluate<ov::element::u64>(ov::as_type_ptr<ov::op::v15::Identity>(node), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled input data type ",
                       node->get_input_element_type(0).get_type_name(),
                       " in evaluate_node().");
    }
}
