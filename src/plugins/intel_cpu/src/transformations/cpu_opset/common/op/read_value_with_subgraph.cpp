// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "read_value_with_subgraph.hpp"

#include "itt.hpp"
#include "transformations/itt.hpp"

ov::intel_cpu::ReadValueWithSubgraph::ReadValueWithSubgraph(const std::shared_ptr<ov::op::util::Variable>& variable,
                                                            const std::shared_ptr<ov::Model>& body) {
    m_variable = variable;
    set_function(body);
}

ov::intel_cpu::ReadValueWithSubgraph::ReadValueWithSubgraph(const std::shared_ptr<ov::op::util::Variable>& variable,
                                                            const std::shared_ptr<ov::Model>& body,
                                                            const OutputVector& args)
    : ReadValueWithSubgraph(variable, body) {
    set_arguments(args);
}

std::string ov::intel_cpu::ReadValueWithSubgraph::get_variable_id() const {
    OPENVINO_ASSERT(m_variable, "Variable is not initialized. Variable_id is unavailable");
    return get_variable()->get_info().variable_id;
}

void ov::intel_cpu::ReadValueWithSubgraph::set_input(const Output<Node>& value,
                                                     const std::shared_ptr<op::v0::Parameter>& body_parameter) {
    OPENVINO_ASSERT(body_parameter != nullptr, "Missing parameter! parameter is is nullptr!");
    auto param_index = m_bodies[0]->get_parameter_index(body_parameter);

    OPENVINO_ASSERT(param_index != -1, "Missing parameter ", body_parameter->get_friendly_name(), " for \'body\'!");

    set_invariant_inputs(value, {body_parameter});
}

ov::Output<ov::Node> ov::intel_cpu::ReadValueWithSubgraph::set_output(
    const std::shared_ptr<op::v0::Result>& body_result) {
    OPENVINO_ASSERT(body_result != nullptr, "Incorrect result in \"body\"! Result cant be \'nullptr\'");
    auto result_id = m_bodies[0]->get_result_index(body_result);

    OPENVINO_ASSERT(result_id != -1, "Missing result ", body_result->get_friendly_name(), "in \'body\'!");

    return set_body_outputs({body_result});
}

std::shared_ptr<ov::Node> ov::intel_cpu::ReadValueWithSubgraph::clone_with_new_inputs(
    const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(intel_cpu_ReadValueWithSubgraphNode_clone_with_new_inputs);

    check_new_args_count(this, new_args);
    auto op =
        std::make_shared<ov::intel_cpu::ReadValueWithSubgraph>(this->get_variable(), get_function()->clone(), new_args);
    OPENVINO_ASSERT(op.get(),
                    op != nullptr,
                    "Cannot clone ",
                    description(),
                    " operation with name ",
                    get_friendly_name());
    op->set_output_size(m_output_descriptions[0].size());
    for (const auto& m_input_descr : m_input_descriptions[0]) {
        op->m_input_descriptions[0].push_back(m_input_descr->copy());
    }
    for (const auto& m_output_descr : m_output_descriptions[0]) {
        op->m_output_descriptions[0].push_back(m_output_descr->copy());
    }
    op->validate_and_infer_types();
    return op;
}

bool ov::intel_cpu::ReadValueWithSubgraph::visit_attributes(AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(intel_cpu_ReadValueWithSubgraphNode_visit_attributes);
    visitor.on_attribute("variable_id", m_variable);

    auto variable_info = m_variable->get_info();
    visitor.on_attribute("variable_type", variable_info.data_type);
    visitor.on_attribute("variable_shape", variable_info.data_shape);
    m_variable->update(variable_info);

    visitor.on_attribute("body", m_bodies[0]);
    visitor.on_attribute("inputs", m_input_descriptions[0]);
    visitor.on_attribute("outputs", m_output_descriptions[0]);
    return true;
}

void ov::intel_cpu::ReadValueWithSubgraph::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(intel_cpu_ReadValueWithSubgraphNode_validate_and_infer_types);

    NODE_VALIDATION_CHECK(this,
                          m_bodies.size() == 1,
                          "ReadValueWithSubgraph contains incorrect number of bodies:",
                          m_bodies.size());

    validate_and_infer_type_body(get_function(), m_input_descriptions[0]);

    auto output_nodes = outputs();

    auto outputs_map = get_mapping_outputs_on_body_description(m_output_descriptions[0]);

    // Checking each output
    for (size_t output_index = 0; output_index < output_nodes.size(); ++output_index) {
        NODE_VALIDATION_CHECK(this,
                              outputs_map.count(output_index) != 0,
                              "Incorrect associating in body! Output ",
                              output_index,
                              " is not associated with results in then_body!");

        auto desc = outputs_map.at(output_index);

        auto node_result = m_bodies[0]->get_results().at(desc->m_body_value_index)->input_value(0);

        set_output_type(output_index, node_result.get_element_type(), node_result.get_partial_shape());
    }
}
