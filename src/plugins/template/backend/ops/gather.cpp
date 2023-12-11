// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/gather.hpp"

#include "evaluate_node.hpp"
#include "utils.hpp"

template <>
bool evaluate_node<ov::op::v8::Gather>(std::shared_ptr<ov::Node> node,
                                       ov::TensorVector& outputs,
                                       const ov::TensorVector& inputs) {
    auto op = ov::as_type_ptr<ov::op::v8::Gather>(node);
    const auto indices = ov::get_tensor_data_as<int64_t>(inputs[1]);

    ov::reference::gather(static_cast<const char*>(inputs[0].data()),
                          indices.data(),
                          static_cast<char*>(outputs[0].data()),
                          op->get_input_shape(0),
                          op->get_input_shape(1),
                          op->get_output_shape(0),
                          op->get_axis(),
                          inputs[0].get_element_type().size(),
                          op->get_batch_dims());
    return true;
}
