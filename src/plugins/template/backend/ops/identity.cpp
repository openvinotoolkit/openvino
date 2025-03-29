// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/identity.hpp"

#include "evaluate_node.hpp"
#include "openvino/core/type/element_iterator.hpp"

template <>
bool evaluate_node<ov::op::v16::Identity>(std::shared_ptr<ov::Node> node,
                                          ov::TensorVector& outputs,
                                          const ov::TensorVector& inputs) {
    const auto input_shape = inputs[0].get_shape();

    outputs[0].set_shape(input_shape);

    ov::reference::identity(static_cast<const char*>(inputs[0].data()),
                            static_cast<char*>(outputs[0].data()),
                            inputs[0].get_byte_size());
    return true;
}
