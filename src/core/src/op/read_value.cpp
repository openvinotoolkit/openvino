// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/read_value.hpp"

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/util/variable_context.hpp"

namespace ov {
namespace op {
namespace v3 {

ReadValue::ReadValue(const Output<Node>& init_value, const std::string& variable_id)
    : ReadValueBase({init_value}),
      m_variable_id(variable_id) {
    constructor_validate_and_infer_types();
}

void ReadValue::validate_and_infer_types() {
    OV_OP_SCOPE(v3_ReadValue_validate_and_infer_types);
    const auto& arg_t = get_input_element_type(0);
    const auto& input_shape = get_input_partial_shape(0);

    util::VariableInfo info = {input_shape, arg_t, m_variable_id};
    if (m_variable == nullptr)
        m_variable = std::make_shared<util::Variable>(info);
    else
        m_variable->update(info);
    set_output_type(0, arg_t, input_shape);
}

std::shared_ptr<Node> ReadValue::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v3_ReadValue_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ReadValue>(new_args.at(0), m_variable_id);
}

bool ReadValue::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v3_ReadValue_visit_attributes);
    visitor.on_attribute("variable_id", m_variable_id);
    return true;
}
}  // namespace v3

namespace v6 {
ReadValue::ReadValue(const std::shared_ptr<util::Variable>& variable) {
    m_variable = variable;
    constructor_validate_and_infer_types();
}

ReadValue::ReadValue(const Output<Node>& init_value, const std::shared_ptr<util::Variable>& variable)
    : ReadValueBase({init_value}) {
    m_variable = variable;
    constructor_validate_and_infer_types();
}

void ReadValue::validate_and_infer_types() {
    OV_OP_SCOPE(v6_ReadValue_validate_and_infer_types);

    OPENVINO_ASSERT(m_variable, "Variable is not initialized.");
    const auto& variable_info = m_variable->get_info();
    const auto& variable_type = variable_info.data_type;
    const auto& variable_shape = variable_info.data_shape;

    // If no inputs provided, it means this ReadValue doesn't have initial subgraph. This is valid.
    if (get_input_size() > 0) {
        const auto& initial_type = get_input_element_type(0);
        const auto& initial_shape = get_input_partial_shape(0);

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
        if (get_input_size() > 0 && variable_info.data_shape.rank().is_dynamic() &&
            variable_info.data_type.is_dynamic()) {
            set_output_type(0, initial_type, initial_shape);
            return;
        }
    }

    set_output_type(0, variable_type, variable_shape);
}

std::shared_ptr<Node> ReadValue::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v6_ReadValue_clone_with_new_inputs);
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

bool ReadValue::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v6_ReadValue_visit_attributes);
    visitor.on_attribute("variable_id", m_variable);

    auto variable_info = m_variable->get_info();
    visitor.on_attribute("variable_type", variable_info.data_type);
    visitor.on_attribute("variable_shape", variable_info.data_shape);
    m_variable->update(variable_info);
    return true;
}

void ReadValue::revalidate_and_infer_types() {
    Node::revalidate_and_infer_types();
}

bool ReadValue::evaluate(TensorVector& outputs,
                         const TensorVector& inputs,
                         const EvaluationContext& evaluation_context) const {
    OV_OP_SCOPE(v6_ReadValue_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);

    const auto& found_context = evaluation_context.find("VariableContext");
    NODE_VALIDATION_CHECK(this, found_context != evaluation_context.end(), "VariableContext not found.");

    const auto& variable_values = found_context->second.as<util::VariableContext>().get_variable_values();
    const auto& var_value = variable_values.find(m_variable);

    const auto use_context = var_value != variable_values.end() && !var_value->second->get_reset();
    auto& output = outputs[0];
    Tensor input;
    if (use_context) {
        input = var_value->second->get_state();
    } else {
        if (!inputs.empty()) {
            input = inputs[0];
        } else {
            auto var_info = m_variable->get_info();
            OPENVINO_ASSERT(var_info.data_shape.is_static() && var_info.data_type.is_static());
            const auto& shape = var_info.data_shape.get_shape();
            const auto& type = var_info.data_type;
            input = ov::Tensor(type, shape);
            memset(input.data(), 0, input.get_byte_size());
        }
    }
    output.set_shape(input.get_shape());
    std::memcpy(output.data(), input.data(), output.get_byte_size());
    return true;
}

bool ReadValue::has_evaluate() const {
    OV_OP_SCOPE(v6_ReadValue_has_evaluate);
    return true;
}

bool ReadValue::can_constant_fold(const OutputVector& input_values) const {
    return false;
}
}  // namespace v6
}  // namespace op
}  // namespace ov
