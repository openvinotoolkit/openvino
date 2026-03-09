// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/fused_conv.hpp"

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/op.hpp"

namespace ov {
namespace op {

FusedConv::FusedConv(const ov::OutputVector& args, const std::shared_ptr<ov::op::util::Variable>& variable)
    : ov::op::Op(args) {
    m_variable = variable;
    constructor_validate_and_infer_types();
}

bool FusedConv::visit_attributes(ov::AttributeVisitor& visitor) {
    OV_OP_SCOPE(FusedConv_visit_attributes);

    visitor.on_attribute("variable_id", m_variable);
    OPENVINO_ASSERT(m_variable, "Variable is not initialized.");

    auto variable_info = m_variable->get_info();
    visitor.on_attribute("variable_type", variable_info.data_type);
    visitor.on_attribute("variable_shape", variable_info.data_shape);
    m_variable->update(variable_info);

    return true;
}

void FusedConv::validate_and_infer_types() {
    OV_OP_SCOPE(FusedConv_validate_and_infer_types);

    NODE_VALIDATION_CHECK(this,
                          get_input_size() == 4,
                          "FusedConv expects 4 inputs, but it has ",
                          get_input_size());
    OPENVINO_ASSERT(m_variable, "Variable is not initialized.");

    // input[0]: [B, conv_dim, S]
    const auto& input_rank = get_input_partial_shape(0).rank();
    NODE_VALIDATION_CHECK(this,
                          input_rank.is_dynamic() || input_rank.get_length() == 3,
                          "Rank of `input` should be 3, but it is ",
                          input_rank);

    // input[1]: [conv_dim, kernel_size]
    const auto& weight_rank = get_input_partial_shape(1).rank();
    NODE_VALIDATION_CHECK(this,
                          weight_rank.is_dynamic() || weight_rank.get_length() == 2,
                          "Rank of `conv_weight` should be 2, but it is ",
                          weight_rank);

    // input[2]: [B] (i32/i64)
    const auto& beam_rank = get_input_partial_shape(2).rank();
    NODE_VALIDATION_CHECK(this,
                          beam_rank.is_dynamic() || beam_rank.get_length() == 1,
                          "Rank of `beam_idx` should be 1, but it is ",
                          beam_rank);

    // input[3]: [B, conv_dim, kernel_size]
    const auto& state_rank = get_input_partial_shape(3).rank();
    NODE_VALIDATION_CHECK(this,
                          state_rank.is_dynamic() || state_rank.get_length() == 3,
                          "Rank of `initial_state` should be 3, but it is ",
                          state_rank);

    const auto& variable_info = m_variable->get_info();
    const auto& variable_shape = variable_info.data_shape;
    const auto& variable_type = variable_info.data_type;
    const auto& initial_shape = get_input_partial_shape(3);
    const auto& initial_type = get_input_element_type(3);

    OPENVINO_ASSERT(variable_shape.relaxes(initial_shape),
                    "The shape specified in the Variable has to extend (relax) the shape inferred from `initial_state`.",
                    " Variable shape: ",
                    variable_shape,
                    " initial_state shape: ",
                    initial_shape);
    OPENVINO_ASSERT(variable_type.is_dynamic() || initial_type == variable_type,
                    "The type specified in the Variable is not compatible with `initial_state`.",
                    " Variable type: ",
                    variable_type,
                    " initial_state type: ",
                    initial_type);

    // output[0]: same shape as input[0] = [B, conv_dim, S]
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    // output[1]: same shape as input[3] = [B, conv_dim, kernel_size]
    set_output_type(1, variable_type, variable_shape);
}

std::shared_ptr<ov::Node> FusedConv::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<FusedConv>(new_args, m_variable);
}

}  // namespace op
}  // namespace ov
