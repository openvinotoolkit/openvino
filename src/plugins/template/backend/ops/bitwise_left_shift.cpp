// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/bitwise_left_shift.hpp"

#include "evaluate_node.hpp"
#include "openvino/reference/bitwise_left_shift.hpp"
#include "utils.hpp"

using namespace ov;

template <element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v15::BitwiseLeftShift>& node,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    OPENVINO_ASSERT(outputs.size() == 1);

    outputs[0].set_shape(infer_broadcast_shape(node.get(), inputs));
    using T = typename ov::element_type_traits<ET>::value_type;
    ov::reference::bitwise_left_shift(inputs[0].data<const T>(),
                                      inputs[1].data<const T>(),
                                      outputs[0].data<T>(),
                                      inputs[0].get_shape(),
                                      inputs[1].get_shape(),
                                      node->get_autob());
    return true;
}

template <>
bool evaluate_node<op::v15::BitwiseLeftShift>(std::shared_ptr<ov::Node> node,
                                              ov::TensorVector& outputs,
                                              const ov::TensorVector& inputs) {
    switch (node->get_input_element_type(0)) {
    case element::u8:
        return evaluate<element::u8>(as_type_ptr<op::v15::BitwiseLeftShift>(node), outputs, inputs);
    case element::i8:
        return evaluate<element::i8>(as_type_ptr<op::v15::BitwiseLeftShift>(node), outputs, inputs);
    case element::u16:
        return evaluate<element::u16>(as_type_ptr<op::v15::BitwiseLeftShift>(node), outputs, inputs);
    case element::i16:
        return evaluate<element::i16>(as_type_ptr<op::v15::BitwiseLeftShift>(node), outputs, inputs);
    case element::u32:
        return evaluate<element::u32>(as_type_ptr<op::v15::BitwiseLeftShift>(node), outputs, inputs);
    case element::i32:
        return evaluate<element::i32>(as_type_ptr<op::v15::BitwiseLeftShift>(node), outputs, inputs);
    case element::u64:
        return evaluate<element::u64>(as_type_ptr<op::v15::BitwiseLeftShift>(node), outputs, inputs);
    case element::i64:
        return evaluate<element::i64>(as_type_ptr<op::v15::BitwiseLeftShift>(node), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", node->get_element_type().get_type_name(), "in evaluate_node()");
    }
}
