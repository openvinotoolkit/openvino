// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/expression_factory.hpp"

#include "snippets/snippets_isa.hpp"

namespace ov {
namespace snippets {
namespace lowered {

void LinearIR::ExpressionFactory::create_expression_inputs(const LinearIR& linear_ir, const ExpressionPtr& expr) {
    OPENVINO_ASSERT(expr != nullptr, "Failed expression inputs creation: expression is null");
    const auto& node = expr->get_node();

    expr->m_input_port_connectors.resize(node->get_input_size(), nullptr);
    for (const auto& input : node->inputs()) {
        const auto input_source = input.get_source_output();
        const auto in_index = input.get_index();
        const auto& parent_expr = linear_ir.get_expr_by_node(input_source.get_node_shared_ptr());
        const auto& port_connector = parent_expr->get_output_port_connector(input_source.get_index());
        port_connector->add_consumer(expr->get_input_port(in_index));
        expr->m_input_port_connectors[in_index] = port_connector;
    }
}

void LinearIR::ExpressionFactory::create_expression_outputs(const ExpressionPtr& expr) {
    OPENVINO_ASSERT(expr != nullptr, "Failed expression outputs creation: expression is null");
    const auto& node = expr->get_node();

    expr->m_output_port_connectors.resize(node->get_output_size(), nullptr);
    for (const auto& output : node->outputs()) {
        const auto out_index = output.get_index();
        const auto source = expr->get_output_port(out_index);
        expr->m_output_port_connectors[out_index] = std::make_shared<PortConnector>(source);
    }
}

// The method verifies of input port connectors to availability of the expression as consumer and add it if missed
void LinearIR::ExpressionFactory::init_expression_inputs(const ExpressionPtr& expr, const std::vector<PortConnectorPtr>& inputs) {
    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& input = inputs[i];
        const auto consumers = input->get_consumers();
        const auto found = std::find_if(consumers.begin(), consumers.end(),
                                        [&](const ExpressionPort& desc) {
                                            return desc.get_index() == i && desc.get_expr() == expr;
                                        });
        if (found == consumers.end()) {
            input->add_consumer(expr->get_input_port(i));
        }
    }
    expr->m_input_port_connectors = inputs;
}

ExpressionPtr LinearIR::ExpressionFactory::create(const std::shared_ptr<ov::op::v0::Parameter>& par,
                                                  const LinearIR& linear_ir, const std::shared_ptr<ov::Model>& model) {
    // Note: ctor of shared_ptr isn't friend class for Expression -> we cannot use directly make_shared<Expression>(args)
    OPENVINO_ASSERT(model != nullptr, "To create IOExpression from Parameter there must be inited model!");
    auto expr = std::shared_ptr<IOExpression>(new IOExpression(par, model->get_parameter_index(par), linear_ir.m_shape_infer_factory));
    create_expression_outputs(expr);
    expr->validate();
    return expr;
}

ExpressionPtr LinearIR::ExpressionFactory::create(const std::shared_ptr<ov::op::v0::Result>& res,
                                                  const LinearIR& linear_ir, const std::shared_ptr<ov::Model>& model) {
    // Note: ctor of shared_ptr isn't friend class for Expression -> we cannot use directly make_shared<Expression>(args)
    OPENVINO_ASSERT(model != nullptr, "To create IOExpression from Result there must be inited model!");
    auto expr = std::shared_ptr<IOExpression>(new IOExpression(res, model->get_result_index(res), linear_ir.m_shape_infer_factory));
    create_expression_inputs(linear_ir, expr);
    // The Result node don't need output port (because of sense of the node). But each node in openvino must have one output at least.
    // The port descriptors are automatically created in constructor. We manually clean output ports.
    expr->m_output_port_descriptors.clear();
    expr->validate();
    return expr;
}

ExpressionPtr LinearIR::ExpressionFactory::create(const std::shared_ptr<ov::Node>& n, const LinearIR& linear_ir,
                                                  const std::shared_ptr<ov::Model>& model) {
    OPENVINO_ASSERT(!ov::is_type<op::LoopBase>(n), "Default expression builder doesn't support LoopBegin and LoopEnd");
    // Note: ctor of shared_ptr isn't friend class for Expression
    auto expr = std::shared_ptr<Expression>(new Expression(n, linear_ir.m_shape_infer_factory));
    create_expression_inputs(linear_ir, expr);
    create_expression_outputs(expr);
    expr->validate();
    return expr;
}

ExpressionPtr LinearIR::ExpressionFactory::create(const std::shared_ptr<op::LoopBegin>& n,
                                                  const std::vector<PortConnectorPtr>& inputs,
                                                  const LinearIR& linear_ir) {
    OPENVINO_ASSERT(inputs.empty(), "LoopBegin cannot have inputs");
    auto expr = std::make_shared<Expression>(Expression(n, linear_ir.m_shape_infer_factory));
    init_expression_inputs(expr, inputs);
    create_expression_outputs(expr);
    expr->validate();
    return expr;
}

ExpressionPtr LinearIR::ExpressionFactory::create(const std::shared_ptr<op::LoopEnd>& n,
                                                  const std::vector<PortConnectorPtr>& inputs,
                                                  const LinearIR& linear_ir) {
    auto expr = std::shared_ptr<Expression>(new Expression(n, linear_ir.m_shape_infer_factory));
    expr->m_input_port_descriptors.resize(inputs.size(), nullptr);
    for (size_t i = 0; i < inputs.size() - 1; ++i) {
        expr->m_input_port_descriptors[i] = std::make_shared<PortDescriptor>();
    }
    const auto& last_input = inputs.back()->get_source();
    OPENVINO_ASSERT(ov::is_type<op::LoopBegin>(last_input.get_expr()->get_node()), "LoopEnd expression expects LoopBegin on last input");
    expr->m_input_port_descriptors[inputs.size() - 1] = last_input.get_descriptor_ptr()->clone();
    init_expression_inputs(expr, inputs);
    // The LoopEnd node don't need output port (because of sense of the node). But each node in openvino must have one output at least.
    // The port descriptors are automatically created in constructor. We manually clean output ports.
    expr->m_output_port_descriptors.clear();
    expr->validate();
    return expr;
}

ExpressionPtr LinearIR::ExpressionFactory::create(const std::shared_ptr<ov::Node>& n,
                                                  const std::vector<PortConnectorPtr>& inputs,
                                                  const LinearIR& linear_ir) {
    OPENVINO_ASSERT(!ov::is_type<ov::op::v0::Parameter>(n) &&
                    !ov::is_type<ov::op::v0::Result>(n),
                    "Expression builder with inputs doesn't support Result and Parameter");
    auto expr = std::shared_ptr<Expression>(new Expression(n, linear_ir.m_shape_infer_factory));
    init_expression_inputs(expr, inputs);
    create_expression_outputs(expr);
    expr->validate();
    // todo: here we blindly synchronize input shapes from parent and child. Remove this when shapes will be stored in
    //  port connector itself
    if (linear_ir.m_shape_infer_factory)
        expr->updateShapes();
    return expr;
}
}// namespace lowered
}// namespace snippets
}// namespace ov
