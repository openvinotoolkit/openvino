// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/assign.hpp"

#include <assign_shape_inference.hpp>

#include "itt.hpp"
#include "ngraph/op/read_value.hpp"
#include "ngraph/op/util/variable.hpp"
#include "ngraph/op/util/variable_context.hpp"
#include "ngraph/ops.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(ov::op::v3::Assign);
BWDCMP_RTTI_DEFINITION(ov::op::v6::Assign);

op::v3::Assign::Assign(const Output<Node>& new_value, const std::string& variable_id)
    : AssignBase({new_value}),
      m_variable_id(variable_id) {
    constructor_validate_and_infer_types();
}

void op::v3::Assign::validate_and_infer_types() {
    OV_OP_SCOPE(v3_Assign_validate_and_infer_types);
    auto value = input_value(0);
    auto arg_t = get_input_element_type(0);
    const auto& input_shape = get_input_partial_shape(0);
    if (!m_variable) {
        NodeVector start_nodes;
        for (const auto& input : inputs()) {
            start_nodes.push_back(input.get_source_output().get_node_shared_ptr());
        }
        auto nodes = topological_sort(start_nodes);
        for (const auto& node : nodes) {
            if (auto read_value = ov::as_type_ptr<op::v3::ReadValue>(node)) {
                if (read_value->get_variable_id() == m_variable_id)
                    m_variable = read_value->get_variable();
            }
        }
        NODE_VALIDATION_CHECK(this, m_variable != nullptr, "Can't find variable with id = ", m_variable_id);
    }
    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape{}};
    std::vector<ov::PartialShape> input_shapes = {input_shape};
    shape_infer(this, input_shapes, output_shapes);
    set_output_type(0, arg_t, output_shapes[0]);
}

shared_ptr<Node> op::v3::Assign::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v3_Assign_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v3::Assign>(new_args.at(0), m_variable_id);
}

bool op::v3::Assign::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v3_Assign_visit_attributes);
    visitor.on_attribute("variable_id", m_variable_id);
    return true;
}

op::v6::Assign::Assign(const Output<Node>& new_value, const std::shared_ptr<Variable>& variable)
    : AssignBase({new_value}) {
    m_variable = variable;
    constructor_validate_and_infer_types();
}

void op::v6::Assign::validate_and_infer_types() {
    OV_OP_SCOPE(v6_Assign_validate_and_infer_types);
    m_variable->update({get_input_partial_shape(0), get_input_element_type(0), m_variable->get_info().variable_id});
    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape{}};
    std::vector<ov::PartialShape> input_shapes = {get_input_partial_shape(0)};
    shape_infer(this, input_shapes, output_shapes);
    set_output_type(0, get_input_element_type(0), output_shapes[0]);
}

shared_ptr<Node> op::v6::Assign::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v6_Assign_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<op::v6::Assign>(new_args.at(0), m_variable);
}

bool op::v6::Assign::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v6_Assign_visit_attributes);
    visitor.on_attribute("variable_id", m_variable);
    return true;
}

bool op::v6::Assign::evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs,
                              const EvaluationContext& evaluation_context) const {
    OV_OP_SCOPE(v6_Assign_evaluate);
    const auto& found_context = evaluation_context.find("VariableContext");
    NODE_VALIDATION_CHECK(this, found_context != evaluation_context.end(), "VariableContext not found.");

    auto& variable_context = const_cast<VariableContext&>(found_context->second.as<VariableContext>());

    const auto& variable_values = variable_context.get_variable_values();

    // automatically allocate memory if not provided by user
    if (variable_values.find(m_variable) == variable_values.end()) {
        auto host_tensor =
            std::make_shared<ngraph::HostTensor>(m_variable->get_info().data_type, m_variable->get_info().data_shape);
        variable_context.set_variable_value(m_variable, make_shared<VariableValue>(host_tensor));
    }

    const auto var_value = variable_values.find(m_variable)->second;
    var_value->set_reset(false);
    const auto& buffer = var_value->get_value();
    buffer->set_unary(inputs[0]);
    outputs[0]->set_unary(inputs[0]);

    void* input = inputs[0]->get_data_ptr();
    outputs[0]->write(input, outputs[0]->get_size_in_bytes());
    buffer->write(input, buffer->get_size_in_bytes());

    return true;
}

bool op::v6::Assign::has_evaluate() const {
    OV_OP_SCOPE(v1_Assign_has_evaluate);
    return true;
}

bool op::v6::Assign::constant_fold(OutputVector& output_values, const OutputVector& inputs_values) {
    return false;
}
