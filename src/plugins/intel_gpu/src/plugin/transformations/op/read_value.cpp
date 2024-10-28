// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/read_value.hpp"
#include "intel_gpu/op/read_values.hpp"
#include "openvino/core/partial_shape.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

ReadValue::ReadValue(const std::shared_ptr<ov::op::util::Variable>& past_values) : Op() {
    m_variable = past_values;
    validate_and_infer_types();
}
ReadValue::ReadValue(const Output<Node>& variable_initializer, const std::shared_ptr<ov::op::util::Variable>& past_values) : Op({variable_initializer}) {
    m_variable = past_values;
    validate_and_infer_types();
}

bool ReadValue::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("variable_id", m_variable);

    auto variable_info = m_variable->get_info();
    visitor.on_attribute("variable_type", variable_info.data_type);
    visitor.on_attribute("variable_shape", variable_info.data_shape);
    m_variable->update(variable_info);
    return true;
}

void ReadValue::validate_and_infer_types(size_t output_idx, const ov::op::util::VariableInfo& variable_info) {
    const auto& variable_type = variable_info.data_type;
    const auto& variable_shape = variable_info.data_shape;

    // If no inputs provided, it means this ReadValue doesn't have initial subgraph. This is valid.
    if (get_input_size() > output_idx) {
        const auto& initial_type = get_input_element_type(output_idx);
        const auto& initial_shape = get_input_partial_shape(output_idx);

        // Variable's shape/type determine a permissible range of values for shape/type inferred from initial_subgraph.
        // If initial_subgraph is set, then we need to check that shape/type inferred from initial_subgraph
        // is within the permissible range.

        const auto compatible_type = variable_type.is_dynamic() || initial_type == variable_type;
        const auto compatible_shape = variable_shape.relaxes(initial_shape);

        OPENVINO_ASSERT(compatible_shape,
                        "The shape specified in the Variable has to extend (relax) the shape "
                        "inferred from the initializing subgraph.",
                        " Variable shape: ",
                        variable_shape,
                        " Initialization shape: ",
                        initial_shape);
        OPENVINO_ASSERT(compatible_type,
                        "The type specified in the Variable has to extend (relax) the type "
                        "inferred from the initializing subgraph.",
                        " Variable type: ",
                        variable_type,
                        " Initialization type: ",
                        initial_type);
        // workaround:
        // dynamic rank/type can be derived from the IRs generated via the prev versions of OV,
        // but dynamic rank/type are not supported in plugins,
        // so we are trying to fix them here using the rank/type of ReadValue 1st input, if it exists
        if (variable_info.data_shape.rank().is_dynamic() && variable_info.data_type.is_dynamic()) {
            set_output_type(output_idx, initial_type, initial_shape);
            return;
        }
    }

    set_output_type(output_idx, variable_type, variable_shape);
}

void ReadValue::validate_and_infer_types() {
    OPENVINO_ASSERT(m_variable, "Variable is not initialized.");
    const auto& variable_info = m_variable->get_info();

    validate_and_infer_types(0, variable_info);
}

std::shared_ptr<Node> ReadValue::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    switch (new_args.size()) {
    case 0:
        return std::make_shared<ReadValue>(m_variable);
    case 1:
        return std::make_shared<ReadValue>(new_args[0], m_variable);
    default:
        OPENVINO_THROW("Unable to clone ReadValue ",
                       this->get_friendly_name(),
                       " Incorrect number of inputs. Expected: 0 or 1. Actual: ",
                       new_args.size());
    }
}

ReadValues::ReadValues(const std::shared_ptr<ov::op::util::Variable>& variable,
                       const std::vector<ov::op::util::VariableInfo>& internal_states_infos)
    : ReadValue(variable)
    , m_internal_states_infos(internal_states_infos) {
    OPENVINO_ASSERT(!internal_states_infos.empty());
    set_output_size(internal_states_infos.size());
    validate_and_infer_types();
}

ReadValues::ReadValues(const std::vector<Output<Node>>& variable_initializers,
                       const std::shared_ptr<ov::op::util::Variable>& variable,
                       const std::vector<ov::op::util::VariableInfo>& internal_states_infos)
    : ReadValue(variable_initializers, variable)
    , m_internal_states_infos(internal_states_infos) {
    OPENVINO_ASSERT(!internal_states_infos.empty());
    set_output_size(internal_states_infos.size());
    validate_and_infer_types();
}

bool ReadValues::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("variable_id", m_variable);

    auto variable_info = m_variable->get_info();
    visitor.on_attribute("variable_type", variable_info.data_type);
    visitor.on_attribute("variable_shape", variable_info.data_shape);
    m_variable->update(variable_info);
    return true;
}

void ReadValues::validate_and_infer_types() {
    OPENVINO_ASSERT(m_variable, "Variable is not initialized.");

    for (size_t i = 0; i < get_output_size(); i++) {
        ReadValue::validate_and_infer_types(i, m_internal_states_infos[i]);
    }
}

std::shared_ptr<Node> ReadValues::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);

    OPENVINO_ASSERT(new_args.empty() || new_args.size() == m_internal_states_infos.size(),
                    "Unable to clone ReadValues op (name=", this->get_friendly_name(), "). ",
                    "Incorrect number of inputs. Expected: 0 or ", m_internal_states_infos.size(), ". ",
                    "Actual: ", new_args.size(), ".");

    if (new_args.size() > 0) {
        return std::make_shared<ReadValues>(new_args, m_variable, m_internal_states_infos);
    } else {
        return std::make_shared<ReadValues>(m_variable, m_internal_states_infos);
    }
}

std::vector<ov::op::util::VariableInfo> ReadValues::get_all_internal_states_info() const {
    return m_internal_states_infos;
}

}  // namespace op
}  // namespace intel_gpu
}  // namespace ov
