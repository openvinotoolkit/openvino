// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/erfinv.hpp"

#include "evaluate_node.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/op/erfinv.hpp"

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v17::ErfInv>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    using T = typename ov::element_type_traits<ET>::value_type;
    outputs[0].set_shape(inputs[0].get_shape());
    ov::reference::erfinv<T>(inputs[0].data<const T>(), outputs[0].data<T>(), ov::shape_size(inputs[0].get_shape()));
    return true;
}

template <>
bool evaluate_node<ov::op::v17::ErfInv>(std::shared_ptr<ov::Node> node,
                                        ov::TensorVector& outputs,
                                        const ov::TensorVector& inputs) {
    switch (node->get_output_element_type(0)) {
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(ov::as_type_ptr<ov::op::v17::ErfInv>(node), outputs, inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(ov::as_type_ptr<ov::op::v17::ErfInv>(node), outputs, inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::v17::ErfInv>(node), outputs, inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64>(ov::as_type_ptr<ov::op::v17::ErfInv>(node), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", node->get_element_type().get_type_name(), " in evaluate_node()");
    }
}
