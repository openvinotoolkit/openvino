// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/divide.hpp"

#include "evaluate_node.hpp"

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v1::Divide>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    using T = ov::fundamental_type_for<ET>;
    ov::reference::divide(inputs[0].data<const T>(),
                          inputs[1].data<const T>(),
                          outputs[0].data<T>(),
                          inputs[0].get_shape(),
                          inputs[1].get_shape(),
                          op->get_autob(),
                          op->is_pythondiv());
    return true;
}

template <>
bool evaluate_node<ov::op::v1::Divide>(std::shared_ptr<ov::Node> node,
                                       ov::TensorVector& outputs,
                                       const ov::TensorVector& inputs) {
    const auto& element_type = node->get_output_element_type(0);

    switch (element_type) {
    case ov::element::i8:
        return evaluate<ov::element::i8>(ov::as_type_ptr<ov::op::v1::Divide>(node), outputs, inputs);
    case ov::element::i16:
        return evaluate<ov::element::i16>(ov::as_type_ptr<ov::op::v1::Divide>(node), outputs, inputs);
    case ov::element::u8:
        return evaluate<ov::element::u8>(ov::as_type_ptr<ov::op::v1::Divide>(node), outputs, inputs);
    case ov::element::u16:
        return evaluate<ov::element::u16>(ov::as_type_ptr<ov::op::v1::Divide>(node), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", element_type, " in evaluate_node()");
    }
}
