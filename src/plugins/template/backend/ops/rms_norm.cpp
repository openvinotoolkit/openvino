// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/rms_norm.hpp"

#include "evaluate_node.hpp"
#include "openvino/core/axis_set.hpp"
#include "openvino/core/rank.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/reference/rms_norm.hpp"
#include "openvino/runtime/tensor.hpp"
#include "utils.hpp"

using namespace ov;

template <element::Type_t T>
bool evaluate(const std::shared_ptr<ov::op::internal::RMSNorm>& node,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    using ET = typename ov::element_type_traits<T>::value_type;
    const auto normalized_axes = ov::util::try_get_normalized_axis_set(inputs[1], inputs[0].get_shape().size(), *node);

    outputs[0].set_shape(inputs[0].get_shape());

    const auto has_scale_input = inputs.size() == 3;
    if (has_scale_input) {
        ov::reference::rms_norm(inputs[0].data<ET>(),
                                normalized_axes,
                                outputs[0].data<ET>(),
                                inputs[0].get_shape(),
                                node->get_epsilon(),
                                inputs[2].get_shape(),
                                inputs[2].data<ET>());
    } else {
        ov::reference::rms_norm(inputs[0].data<ET>(),
                                normalized_axes,
                                outputs[0].data<ET>(),
                                inputs[0].get_shape(),
                                node->get_epsilon());
    }
    return true;
}

template <>
bool evaluate_node<op::internal::RMSNorm>(std::shared_ptr<ov::Node> node,
                                          ov::TensorVector& outputs,
                                          const ov::TensorVector& inputs) {
    switch (node->get_input_element_type(0)) {
    case element::bf16:
        return evaluate<element::bf16>(as_type_ptr<op::internal::RMSNorm>(node), outputs, inputs);
    case element::f16:
        return evaluate<element::f16>(as_type_ptr<op::internal::RMSNorm>(node), outputs, inputs);
    case element::f64:
        return evaluate<element::f64>(as_type_ptr<op::internal::RMSNorm>(node), outputs, inputs);
    case element::f32:
        return evaluate<element::f32>(as_type_ptr<op::internal::RMSNorm>(node), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", node->get_element_type().get_type_name(), " in evaluate_node()");
    }
}
