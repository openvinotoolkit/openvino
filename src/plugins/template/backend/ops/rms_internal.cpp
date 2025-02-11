// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluate_node.hpp"
#include "openvino/core/axis_set.hpp"
#include "openvino/core/rank.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/reference/rms_norm.hpp"
#include "openvino/runtime/tensor.hpp"
#include "ov_ops/rms.hpp"
#include "utils.hpp"

using namespace ov;

template <element::Type_t T>
bool evaluate(const std::shared_ptr<ov::op::internal::RMS>& node,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    using ET = typename ov::element_type_traits<T>::value_type;
    const auto normalized_axes = AxisSet{ov::util::normalize_axis(-1, inputs[0].get_shape().size())};

    outputs[0].set_shape(inputs[0].get_shape());

    const auto& in_type = inputs[0].get_element_type();
    const auto& out_type = outputs[0].get_element_type();

    // The type compression mechanism is implemented for F16 only
    // The scale is expected to have the same type as the first input
    if (in_type != out_type && out_type == ov::element::f16) {
        ov::reference::rms_norm_mul_convert_out(inputs[0].data<ET>(),
                                                normalized_axes,
                                                outputs[0].data<ov::float16>(),
                                                inputs[0].get_shape(),
                                                node->get_epsilon(),
                                                inputs[1].get_shape(),
                                                inputs[1].data<ET>());

    } else {
        ov::reference::rms_norm(inputs[0].data<ET>(),
                                normalized_axes,
                                outputs[0].data<ET>(),
                                inputs[0].get_shape(),
                                node->get_epsilon(),
                                inputs[1].get_shape(),
                                inputs[1].data<ET>());
    }
    return true;
}

template <>
bool evaluate_node<op::internal::RMS>(std::shared_ptr<ov::Node> node,
                                      ov::TensorVector& outputs,
                                      const ov::TensorVector& inputs) {
    switch (node->get_input_element_type(0)) {
    case element::bf16:
        return evaluate<element::bf16>(as_type_ptr<op::internal::RMS>(node), outputs, inputs);
    case element::f16:
        return evaluate<element::f16>(as_type_ptr<op::internal::RMS>(node), outputs, inputs);
    case element::f64:
        return evaluate<element::f64>(as_type_ptr<op::internal::RMS>(node), outputs, inputs);
    case element::f32:
        return evaluate<element::f32>(as_type_ptr<op::internal::RMS>(node), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", node->get_input_element_type(0).get_type_name(), " in evaluate_node()");
    }
}
