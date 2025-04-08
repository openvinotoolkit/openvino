// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/group_normalization.hpp"

#include "evaluate_node.hpp"
#include "openvino/op/group_normalization.hpp"

using namespace ov;

template <element::Type_t T>
bool evaluate(const std::shared_ptr<ov::op::v12::GroupNormalization>& node,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    using ET = typename ov::element_type_traits<T>::value_type;
    outputs[0].set_shape(inputs[0].get_shape());
    ov::reference::group_normalization(inputs[0].data<ET>(),
                                       inputs[1].data<ET>(),
                                       inputs[2].data<ET>(),
                                       outputs[0].data<ET>(),
                                       inputs[0].get_shape(),
                                       static_cast<size_t>(node->get_num_groups()),
                                       node->get_epsilon());
    return true;
}

template <>
bool evaluate_node<op::v12::GroupNormalization>(std::shared_ptr<ov::Node> node,
                                                ov::TensorVector& outputs,
                                                const ov::TensorVector& inputs) {
    switch (node->get_input_element_type(0)) {
    case element::bf16:
        return evaluate<element::bf16>(as_type_ptr<op::v12::GroupNormalization>(node), outputs, inputs);
    case element::f16:
        return evaluate<element::f16>(as_type_ptr<op::v12::GroupNormalization>(node), outputs, inputs);
    case element::f64:
        return evaluate<element::f64>(as_type_ptr<op::v12::GroupNormalization>(node), outputs, inputs);
    case element::f32:
        return evaluate<element::f32>(as_type_ptr<op::v12::GroupNormalization>(node), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", node->get_element_type().get_type_name(), " in evaluate_node()");
    }
}
