// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/atan2.hpp"

#include "evaluate_node.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/op/atan2.hpp"

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v17::Atan2>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    using T = ov::fundamental_type_for<ET>;
    outputs[0].set_shape(infer_broadcast_shape(op.get(), inputs));
    ov::reference::atan2(inputs[0].data<const T>(),
                         inputs[1].data<const T>(),
                         outputs[0].data<T>(),
                         inputs[0].get_shape(),
                         inputs[1].get_shape(),
                         op->get_autob());
    return true;
}

template <>
bool evaluate_node<ov::op::v17::Atan2>(std::shared_ptr<ov::Node> node,
                                        ov::TensorVector& outputs,
                                        const ov::TensorVector& inputs) {
    const auto& element_type = node->get_output_element_type(0);

    switch (element_type) {
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(ov::as_type_ptr<ov::op::v17::Atan2>(node), outputs, inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(ov::as_type_ptr<ov::op::v17::Atan2>(node), outputs, inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::v17::Atan2>(node), outputs, inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64>(ov::as_type_ptr<ov::op::v17::Atan2>(node), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", element_type, " in evaluate_node()");
    }
}
