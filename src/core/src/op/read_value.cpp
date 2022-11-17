// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/read_value.hpp"

#include <read_value_shape_inference.hpp>

#include "itt.hpp"
#include "ngraph/op/util/variable_context.hpp"
#include "ngraph/ops.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(ov::op::v3::ReadValue);
BWDCMP_RTTI_DEFINITION(ov::op::v6::ReadValue);

op::v3::ReadValue::ReadValue(const Output<Node>& init_value, const std::string& variable_id)
    : ReadValueBase({init_value}),
      m_variable_id(variable_id) {
    constructor_validate_and_infer_types();
}

void op::v3::ReadValue::validate_and_infer_types() {
    OV_OP_SCOPE(v3_ReadValue_validate_and_infer_types);
    auto arg_t = get_input_element_type(0);
    auto input_shape = get_input_partial_shape(0);

    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape{}};
    std::vector<ov::PartialShape> input_shapes = {input_shape};
    shape_infer(this, input_shapes, output_shapes);

    const auto& output_shape = output_shapes[0];
    VariableInfo info = {output_shape, arg_t, m_variable_id};
    if (m_variable == nullptr)
        m_variable = std::make_shared<Variable>(info);
    else
        m_variable->update(info);
    set_output_type(0, arg_t, output_shape);
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

op::v6::ReadValue::ReadValue(const Output<Node>& init_value, const shared_ptr<Variable>& variable)
    : ReadValueBase({init_value}) {
    m_variable = variable;
    constructor_validate_and_infer_types();
}

void op::v6::ReadValue::validate_and_infer_types() {
    OV_OP_SCOPE(v6_ReadValue_validate_and_infer_types);
    const auto arg_t = get_input_element_type(0);
    auto input_shape = get_input_partial_shape(0);
    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape{}};
    std::vector<ov::PartialShape> input_shapes = {input_shape};
    shape_infer(this, input_shapes, output_shapes);
    const auto& output_shape = output_shapes[0];
    NGRAPH_CHECK(m_variable, "Variable is not initialized.");
    VariableInfo var_info = {output_shape, element::dynamic, m_variable->get_info().variable_id};
    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(var_info.data_type, m_variable->get_info().data_type, arg_t),
                          "Variables types are inconsistent.");
    NODE_VALIDATION_CHECK(this,
                          ov::PartialShape::merge_into(var_info.data_shape, m_variable->get_info().data_shape),
                          "Variable shape and output shape are inconsistent.");
    m_variable->update(var_info);
    set_output_type(0, arg_t, output_shape);
}

shared_ptr<Node> op::v6::ReadValue::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v6_ReadValue_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<ReadValue>(new_args.at(0), m_variable);
}

bool op::v6::ReadValue::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v6_ReadValue_visit_attributes);
    visitor.on_attribute("variable_id", m_variable);
    return true;
}

void op::v6::ReadValue::revalidate_and_infer_types() {
    VariableInfo var_info{ov::PartialShape::dynamic(), element::dynamic, m_variable->get_info().variable_id};
    m_variable->update(var_info);
    Node::revalidate_and_infer_types();
}

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
    auto zero_tensor = make_shared<HostTensor>(zero_const);
    const auto& input_tensor = use_context ? var_value->second->get_value() : zero_tensor;
    outputs[0]->set_unary(input_tensor);

    void* input = input_tensor->get_data_ptr();
    outputs[0]->write(input, outputs[0]->get_size_in_bytes());
    return true;
}

bool op::v6::ReadValue::has_evaluate() const {
    OV_OP_SCOPE(v6_ReadValue_has_evaluate);
    return true;
}

bool op::v6::ReadValue::constant_fold(OutputVector& output_values, const OutputVector& inputs_values) {
    return false;
}
