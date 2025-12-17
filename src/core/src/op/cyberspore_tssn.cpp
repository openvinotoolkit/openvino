// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/cyberspore_tssn.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace op {
namespace v0 {

CybersporeTSSN::CybersporeTSSN(const Output<Node>& input_events,
                               const Output<Node>& state_matrix,
                               const Output<Node>& selective_params,
                               float homeostatic_setpoint,
                               float decay_rate)
    : Op({input_events, state_matrix, selective_params}),
      m_homeostatic_setpoint(homeostatic_setpoint),
      m_decay_rate(decay_rate) {
    constructor_validate_and_infer_types();
}

void CybersporeTSSN::validate_and_infer_types() {
    // Check input counts
    NODE_VALIDATION_CHECK(this, get_input_size() == 3, "CybersporeTSSN expects 3 inputs.");

    const auto& input_events_type = get_input_element_type(0);
    const auto& state_matrix_type = get_input_element_type(1);
    const auto& selective_params_type = get_input_element_type(2);

    // Validate input types
    // Note: strict type checking might be too restrictive if we want to allow casting or other types.
    // However, the requirement specified Input_Events (Ternary), State_Matrix (Ternary).
    // Since we added t2, we check for it.
    // For robustness, we can also allow standard integer/float types if t2 is not strictly enforced at all times.
    
    NODE_VALIDATION_CHECK(this, 
                          input_events_type == element::t2 || input_events_type.is_dynamic(),
                          "Input Events must be of type t2 (Ternary). Got: ", input_events_type);
    
    NODE_VALIDATION_CHECK(this, 
                          state_matrix_type == element::t2 || state_matrix_type.is_dynamic(),
                          "State Matrix must be of type t2 (Ternary). Got: ", state_matrix_type);

    NODE_VALIDATION_CHECK(this, 
                          selective_params_type.is_real() || selective_params_type.is_dynamic(),
                          "Selective Params must be a real type (Float/BF16). Got: ", selective_params_type);

    // Shapes validation
    // Assuming simple SSM logic where output shape matches input_events or state_matrix
    // Let's assume output has the same shape as State_Matrix (ht)
    const auto& state_matrix_shape = get_input_partial_shape(1);
    
    // The output is the new state ht, so it should have the same type as State_Matrix (t2)
    set_output_type(0, state_matrix_type, state_matrix_shape);
}

std::shared_ptr<Node> CybersporeTSSN::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<CybersporeTSSN>(new_args.at(0),
                                            new_args.at(1),
                                            new_args.at(2),
                                            m_homeostatic_setpoint,
                                            m_decay_rate);
}

bool CybersporeTSSN::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("homeostatic_setpoint", m_homeostatic_setpoint);
    visitor.on_attribute("decay_rate", m_decay_rate);
    return true;
}

}  // namespace v0
}  // namespace op
}  // namespace ov
