// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/histc.hpp"

#include "evaluate_node.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/op/histc.hpp"

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v17::Histc>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    using T = ov::fundamental_type_for<ET>;
    outputs[0].set_shape(op->get_output_shape(0));
    ov::reference::histc(inputs[0].data<const T>(),
                         outputs[0].data<T>(),
                         inputs[0].get_size(),
                         outputs[0].get_size(),
                         op->get_min(),
                         op->get_max());
    return true;
}

template <>
bool evaluate_node<ov::op::v17::Histc>(std::shared_ptr<ov::Node> node,
                                       ov::TensorVector& outputs,
                                       const ov::TensorVector& inputs) {
    const auto op = ov::as_type_ptr<ov::op::v17::Histc>(node);
    switch (node->get_input_element_type(0)) {
    case ov::element::f16:
        return evaluate<ov::element::f16>(op, outputs, inputs);
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(op, outputs, inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(op, outputs, inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64>(op, outputs, inputs);
    case ov::element::i8:
        return evaluate<ov::element::i8>(op, outputs, inputs);
    case ov::element::i16:
        return evaluate<ov::element::i16>(op, outputs, inputs);
    case ov::element::i32:
        return evaluate<ov::element::i32>(op, outputs, inputs);
    case ov::element::i64:
        return evaluate<ov::element::i64>(op, outputs, inputs);
    case ov::element::u8:
        return evaluate<ov::element::u8>(op, outputs, inputs);
    case ov::element::u16:
        return evaluate<ov::element::u16>(op, outputs, inputs);
    case ov::element::u32:
        return evaluate<ov::element::u32>(op, outputs, inputs);
    case ov::element::u64:
        return evaluate<ov::element::u64>(op, outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", node->get_input_element_type(0), " in evaluate_node()");
    }
}
