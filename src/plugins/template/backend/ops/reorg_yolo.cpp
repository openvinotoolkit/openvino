// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/reorg_yolo.hpp"

#include "evaluate_node.hpp"

bool evaluate(const std::shared_ptr<ov::op::v0::ReorgYolo>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    ov::reference::reorg_yolo(static_cast<char*>(inputs[0].data()),
                              static_cast<char*>(outputs[0].data()),
                              inputs[0].get_shape(),
                              op->get_strides().at(0),
                              inputs[0].get_element_type().size());
    return true;
}

template <>
bool evaluate_node<ov::op::v0::ReorgYolo>(std::shared_ptr<ov::Node> node,
                                          ov::TensorVector& outputs,
                                          const ov::TensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ov::op::v1::Select>(node) || ov::is_type<ov::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    return evaluate(ov::as_type_ptr<ov::op::v0::ReorgYolo>(node), outputs, inputs);
}
