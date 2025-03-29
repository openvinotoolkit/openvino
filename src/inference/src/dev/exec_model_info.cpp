// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/exec_model_info.hpp"

ov::exec_model_info::ExecutionNode::ExecutionNode() {}

ov::exec_model_info::ExecutionNode::ExecutionNode(const ov::OutputVector& arguments, size_t output_size)
    : ov::op::Op() {
    set_arguments(arguments);
    set_output_size(output_size);
}

std::shared_ptr<ov::Node> ov::exec_model_info::ExecutionNode::clone_with_new_inputs(
    const ov::OutputVector& inputs) const {
    auto cloned = std::make_shared<ExecutionNode>();

    cloned->set_arguments(inputs);

    for (auto kvp : get_rt_info())
        cloned->get_rt_info()[kvp.first] = kvp.second;

    for (size_t i = 0; i < get_output_size(); ++i)
        cloned->set_output_type(i, get_output_element_type(i), get_output_partial_shape(i));

    return cloned;
}

/**
 * @brief      Visits attributes of the node
 *
 * @param[in]  visitor  An attribute visitor
 *
 * @return     Returns `true` if an operation has completed successfully
 */
bool ov::exec_model_info::ExecutionNode::visit_attributes(ov::AttributeVisitor& /*visitor*/) {
    return true;
}
