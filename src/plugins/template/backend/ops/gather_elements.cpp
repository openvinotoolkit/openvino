// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/gather_elements.hpp"

#include "evaluate_node.hpp"

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v6::GatherElements>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    using T = typename ov::element_type_traits<ET>::value_type;
    ov::Shape params_shape = inputs[0].get_shape();
    ov::Shape indices_shape = inputs[1].get_shape();

    outputs[0].set_shape(indices_shape);

    if (inputs[1].get_element_type() == ov::element::i64) {
        ov::reference::gather_elements<T, int64_t>(inputs[0].data<T>(),
                                                   inputs[1].data<int64_t>(),
                                                   outputs[0].data<T>(),
                                                   inputs[0].get_shape(),
                                                   inputs[1].get_shape(),
                                                   outputs[0].get_shape(),
                                                   op->get_axis());
    } else if (inputs[1].get_element_type() == ov::element::i32) {
        ov::reference::gather_elements<T, int32_t>(inputs[0].data<T>(),
                                                   inputs[1].data<int32_t>(),
                                                   outputs[0].data<T>(),
                                                   inputs[0].get_shape(),
                                                   inputs[1].get_shape(),
                                                   outputs[0].get_shape(),
                                                   op->get_axis());
    } else {
        OPENVINO_THROW("Unexpected indices type");
    }

    return true;
}

template <>
bool evaluate_node<ov::op::v6::GatherElements>(std::shared_ptr<ov::Node> node,
                                               ov::TensorVector& outputs,
                                               const ov::TensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ov::op::v1::Select>(node) || ov::is_type<ov::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ov::element::boolean:
        return evaluate<ov::element::boolean>(ov::as_type_ptr<ov::op::v6::GatherElements>(node), outputs, inputs);
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(ov::as_type_ptr<ov::op::v6::GatherElements>(node), outputs, inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(ov::as_type_ptr<ov::op::v6::GatherElements>(node), outputs, inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64>(ov::as_type_ptr<ov::op::v6::GatherElements>(node), outputs, inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::v6::GatherElements>(node), outputs, inputs);
    case ov::element::i4:
        return evaluate<ov::element::i4>(ov::as_type_ptr<ov::op::v6::GatherElements>(node), outputs, inputs);
    case ov::element::i8:
        return evaluate<ov::element::i8>(ov::as_type_ptr<ov::op::v6::GatherElements>(node), outputs, inputs);
    case ov::element::i16:
        return evaluate<ov::element::i16>(ov::as_type_ptr<ov::op::v6::GatherElements>(node), outputs, inputs);
    case ov::element::i32:
        return evaluate<ov::element::i32>(ov::as_type_ptr<ov::op::v6::GatherElements>(node), outputs, inputs);
    case ov::element::i64:
        return evaluate<ov::element::i64>(ov::as_type_ptr<ov::op::v6::GatherElements>(node), outputs, inputs);
    case ov::element::u1:
        return evaluate<ov::element::u1>(ov::as_type_ptr<ov::op::v6::GatherElements>(node), outputs, inputs);
    case ov::element::u4:
        return evaluate<ov::element::u4>(ov::as_type_ptr<ov::op::v6::GatherElements>(node), outputs, inputs);
    case ov::element::u8:
        return evaluate<ov::element::u8>(ov::as_type_ptr<ov::op::v6::GatherElements>(node), outputs, inputs);
    case ov::element::u16:
        return evaluate<ov::element::u16>(ov::as_type_ptr<ov::op::v6::GatherElements>(node), outputs, inputs);
    case ov::element::u32:
        return evaluate<ov::element::u32>(ov::as_type_ptr<ov::op::v6::GatherElements>(node), outputs, inputs);
    case ov::element::u64:
        return evaluate<ov::element::u64>(ov::as_type_ptr<ov::op::v6::GatherElements>(node), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", node->get_element_type().get_type_name(), " in evaluate_node()");
    }
}
