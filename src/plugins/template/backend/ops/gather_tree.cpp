// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/gather_tree.hpp"

#include "evaluate_node.hpp"

bool evaluate(const std::shared_ptr<ov::op::v1::GatherTree>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    ov::reference::gather_tree(static_cast<const char*>(inputs[0].data()),
                               static_cast<const char*>(inputs[1].data()),
                               static_cast<const char*>(inputs[2].data()),
                               static_cast<const char*>(inputs[3].data()),
                               static_cast<char*>(outputs[0].data()),
                               op->get_input_shape(0),
                               op->get_input_shape(1),
                               op->get_input_shape(2),
                               op->get_input_shape(3),
                               inputs[1].get_element_type());
    return true;
}

template <>
bool evaluate_node<ov::op::v1::GatherTree>(std::shared_ptr<ov::Node> node,
                                           ov::TensorVector& outputs,
                                           const ov::TensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ov::op::v1::Select>(node) || ov::is_type<ov::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    return evaluate(ov::as_type_ptr<ov::op::v1::GatherTree>(node), outputs, inputs);
}
