// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/bitwise_not.hpp"

#include "evaluate_node.hpp"
#include "openvino/reference/not.hpp"

using namespace ov;

template <element::Type_t T>
bool evaluate(const std::shared_ptr<ov::op::v13::BitwiseNot>& node,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    using ET = typename ov::element_type_traits<T>::value_type;
    if (T == element::Type_t::boolean) {
        ov::reference::logical_not(inputs[0].data<ET>(), outputs[0].data<ET>(), shape_size(inputs[0].get_shape()));
    } else {
        ov::reference::bitwise_not(inputs[0].data<ET>(), outputs[0].data<ET>(), shape_size(inputs[0].get_shape()));
    }
    return true;
}

template <>
bool evaluate_node<op::v13::BitwiseNot>(std::shared_ptr<ov::Node> node,
                                        ov::TensorVector& outputs,
                                        const ov::TensorVector& inputs) {
    switch (node->get_input_element_type(0)) {
    case element::boolean:
        return evaluate<element::boolean>(as_type_ptr<op::v13::BitwiseNot>(node), outputs, inputs);
    case element::u8:
        return evaluate<element::u8>(as_type_ptr<op::v13::BitwiseNot>(node), outputs, inputs);
    case element::i8:
        return evaluate<element::i8>(as_type_ptr<op::v13::BitwiseNot>(node), outputs, inputs);
    case element::u16:
        return evaluate<element::u16>(as_type_ptr<op::v13::BitwiseNot>(node), outputs, inputs);
    case element::i16:
        return evaluate<element::i16>(as_type_ptr<op::v13::BitwiseNot>(node), outputs, inputs);
    case element::u32:
        return evaluate<element::u32>(as_type_ptr<op::v13::BitwiseNot>(node), outputs, inputs);
    case element::i32:
        return evaluate<element::i32>(as_type_ptr<op::v13::BitwiseNot>(node), outputs, inputs);
    case element::u64:
        return evaluate<element::u64>(as_type_ptr<op::v13::BitwiseNot>(node), outputs, inputs);
    case element::i64:
        return evaluate<element::i64>(as_type_ptr<op::v13::BitwiseNot>(node), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", node->get_element_type().get_type_name(), "in evaluate_node()");
    }
}
