// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/reduce_mean.hpp"

#include "evaluate_node.hpp"

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v1::ReduceMean>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    using T = ov::fundamental_type_for<ET>;
    ov::reference::reduce_mean(inputs[0].data<const T>(),
                               outputs[0].data<T>(),
                               inputs[0].get_shape(),
                               op->get_reduction_axes());
    return true;
}

template <>
bool evaluate_node<ov::op::v1::ReduceMean>(std::shared_ptr<ov::Node> node,
                                           ov::TensorVector& outputs,
                                           const ov::TensorVector& inputs) {
    const auto& element_type = node->get_output_element_type(0);

    switch (element_type) {
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(ov::as_type_ptr<ov::op::v1::ReduceMean>(node), outputs, inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(ov::as_type_ptr<ov::op::v1::ReduceMean>(node), outputs, inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::v1::ReduceMean>(node), outputs, inputs);
    case ov::element::i8:
        return evaluate<ov::element::i8>(ov::as_type_ptr<ov::op::v1::ReduceMean>(node), outputs, inputs);
    case ov::element::u8:
        return evaluate<ov::element::u8>(ov::as_type_ptr<ov::op::v1::ReduceMean>(node), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", element_type, " in evaluate_node()");
    }
}
