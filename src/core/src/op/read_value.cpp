// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/read_value.hpp"

#include "itt.hpp"
#include "ngraph/op/util/variable_context.hpp"
#include "ngraph/ops.hpp"

using namespace std;
using namespace ngraph;

op::v3::ReadValue::ReadValue(const Output<Node>& init_value, const std::string& variable_id)
    : ReadValueBase({init_value}),
      m_variable_id(variable_id) {
    constructor_validate_and_infer_types();
}

void op::v3::ReadValue::validate_and_infer_types() {
    OV_OP_SCOPE(v3_ReadValue_validate_and_infer_types);
    auto arg_t = get_input_element_type(0);
    const auto& input_shape = get_input_partial_shape(0);

    VariableInfo info = {input_shape, arg_t, m_variable_id};
    if (m_variable == nullptr)
        m_variable = std::make_shared<Variable>(info);
    else
        m_variable->update(info);
    set_output_type(0, arg_t, input_shape);
}

shared_ptr<Node> op::v3::ReadValue::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v3_ReadValue_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<ReadValue>(new_args.at(0), m_variable_id);
}

bool op::v3::ReadValue::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v3_ReadValue_visit_attributes);
    visitor.on_attribute("variable_id", m_variable_id);
    return true;
}

op::v6::ReadValue::ReadValue(const shared_ptr<Variable>& variable) {
    m_variable = variable;
    constructor_validate_and_infer_types();
}

op::v6::ReadValue::ReadValue(const Output<Node>& init_value, const shared_ptr<Variable>& variable)
    : ReadValueBase({init_value}) {
    m_variable = variable;
    constructor_validate_and_infer_types();
}

void op::v6::ReadValue::validate_and_infer_types() {
    OV_OP_SCOPE(v6_ReadValue_validate_and_infer_types);

    OPENVINO_ASSERT(m_variable, "Variable is not initialized.");
    auto variable_info = m_variable->get_info();
    auto variable_type = variable_info.data_type;
    auto variable_shape = variable_info.data_shape;

    // If no inputs provided, it means this ReadValue doesn't have initial subgraph. This is valid.
    if (get_input_size() > 0) {
        const auto initial_type = get_input_element_type(0);
        const auto& initial_shape = get_input_partial_shape(0);

        // Variable shape/type determines a permissible range of values for shape/type inferred from initial_subgraph.
        // If initial_subgraph is set, then we need to check that shape/type inferred from initial_subgraph
        // is within the permissible range.

        bool compatible_type = variable_type.is_dynamic() || initial_type == variable_type;
        bool compatible_shape = variable_shape.rank().relaxes(initial_shape.rank());

        if (compatible_shape && initial_shape.rank().is_static() && variable_shape.rank().is_static()) {
            OPENVINO_ASSERT(initial_shape.rank().get_length() == variable_shape.rank().get_length(),
                            "Ranks of initial_shape and variable_shape do not match.");
            for (int64_t i = 0; i < variable_shape.rank().get_length(); ++i) {
                compatible_shape = compatible_shape && variable_shape[i].relaxes(initial_shape[i]);
            }
        }
        OPENVINO_ASSERT(compatible_shape,
                        "The shape specified in the Variable have to extend (relax) the shape "
                        "inferred from the initializing subgraph.");
        OPENVINO_ASSERT(compatible_type,
                        "The type specified in the Variable have to extend (relax) the type "
                        "inferred from the initializing subgraph.");
    }

    set_output_type(0, variable_type, variable_shape);
}

shared_ptr<Node> op::v6::ReadValue::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v6_ReadValue_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<ReadValue>(new_args.at(0), m_variable);
}

bool op::v6::ReadValue::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v6_ReadValue_visit_attributes);
    visitor.on_attribute("variable_id", m_variable);

    auto variable_info = m_variable->get_info();
    visitor.on_attribute("variable_type", variable_info.data_type);
    visitor.on_attribute("variable_shape", variable_info.data_shape);
    m_variable->update(variable_info);
    return true;
}

void op::v6::ReadValue::revalidate_and_infer_types() {
    Node::revalidate_and_infer_types();
}

OPENVINO_SUPPRESS_DEPRECATED_START
bool op::v6::ReadValue::evaluate(const HostTensorVector& outputs,
                                 const HostTensorVector& inputs,
                                 const EvaluationContext& evaluation_context) const {
    OV_OP_SCOPE(v6_ReadValue_evaluate);
    const auto& found_context = evaluation_context.find("VariableContext");
    NODE_VALIDATION_CHECK(this, found_context != evaluation_context.end(), "VariableContext not found.");

    const auto& variable_values = found_context->second.as<VariableContext>().get_variable_values();
    const auto& var_value = variable_values.find(m_variable);

    bool use_context = var_value != variable_values.end() && !var_value->second->get_reset();

    // initial value (inputs[0]) is not supported, use zeros
    auto zero_const = make_shared<v0::Constant>(inputs[0]->get_element_type(), inputs[0]->get_shape(), 0);
    OPENVINO_SUPPRESS_DEPRECATED_START
    auto zero_tensor = make_shared<HostTensor>(zero_const);
    const auto& input_tensor = use_context ? var_value->second->get_value() : zero_tensor;
    OPENVINO_SUPPRESS_DEPRECATED_END
    outputs[0]->set_unary(input_tensor);

    void* input = input_tensor->get_data_ptr();
    outputs[0]->write(input, outputs[0]->get_size_in_bytes());
    return true;
}
OPENVINO_SUPPRESS_DEPRECATED_END

bool op::v6::ReadValue::has_evaluate() const {
    OV_OP_SCOPE(v6_ReadValue_has_evaluate);
    return true;
}

bool op::v6::ReadValue::constant_fold(OutputVector& output_values, const OutputVector& inputs_values) {
    return false;
}
