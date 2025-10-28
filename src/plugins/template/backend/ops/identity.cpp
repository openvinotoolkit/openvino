// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/identity.hpp"

#include "evaluate_node.hpp"
#include "openvino/core/type/element_iterator.hpp"
#include "openvino/op/identity.hpp"

template <>
bool evaluate_node<ov::op::v16::Identity>(std::shared_ptr<ov::Node> node,
                                          ov::TensorVector& outputs,
                                          const ov::TensorVector& inputs) {
    const auto input_shape = inputs[0].get_shape();

    outputs[0].set_shape(input_shape);
    auto element_type = node->get_element_type();

    if (element_type == ov::element::string) {
        ov::reference::identity(inputs[0].data<const std::string>(),
                                outputs[0].data<std::string>(),
                                inputs[0].get_size());

    } else {
        ov::reference::identity(inputs[0].data<const char>(), outputs[0].data<char>(), inputs[0].get_byte_size());
    }

    return true;
}
