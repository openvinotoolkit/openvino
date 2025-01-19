// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/expression_factory.hpp"

#include "snippets/snippets_isa.hpp"

namespace ov {
namespace snippets {
namespace lowered {

template<>
std::shared_ptr<Expression> ExpressionFactory::build(const std::shared_ptr<Node>& n, const std::vector<PortConnectorPtr>& inputs) {
    if (const auto par = ov::as_type_ptr<ov::op::v0::Parameter>(n)) {
        return create(par, inputs, m_shape_infer_factory);
    } else if (const auto res = ov::as_type_ptr<ov::op::v0::Result>(n)) {
        return create(res, inputs, m_shape_infer_factory);
    } else if (const auto loop_begin = ov::as_type_ptr<op::LoopBegin>(n)) {
        return create(loop_begin, inputs, m_shape_infer_factory);
    } else if (const auto loop_end = ov::as_type_ptr<op::LoopEnd>(n)) {
        return create(loop_end, inputs, m_shape_infer_factory);
    } else if (const auto spill_begin = ov::as_type_ptr<op::RegSpillBegin>(n)) {
        return create(spill_begin, inputs, m_shape_infer_factory);
    } else if (const auto spill_end = ov::as_type_ptr<op::RegSpillEnd>(n)) {
        return create(spill_end, inputs, m_shape_infer_factory);
    } else if (const auto buffer = ov::as_type_ptr<op::Buffer>(n)) {
        return create<BufferExpression>(buffer, inputs, m_shape_infer_factory);
#ifdef SNIPPETS_DEBUG_CAPS
    } else if (const auto perf_counter = ov::as_type_ptr<op::PerfCountBeginBase>(n)) {
        return create(perf_counter, inputs, m_shape_infer_factory);
    } else if (const auto perf_counter = ov::as_type_ptr<op::PerfCountEndBase>(n)) {
        return create(perf_counter, inputs, m_shape_infer_factory);
#endif
    }
    return create(n, inputs, m_shape_infer_factory);
}

void ExpressionFactory::create_expression_outputs(const ExpressionPtr& expr) {
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
void ExpressionFactory::init_expression_inputs(const ExpressionPtr& expr, const std::vector<PortConnectorPtr>& inputs) {
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

ExpressionPtr ExpressionFactory::create(const std::shared_ptr<ov::op::v0::Parameter>& par, const std::vector<PortConnectorPtr>& inputs,
                                        const std::shared_ptr<IShapeInferSnippetsFactory>& shape_infer_factory) {
    OPENVINO_ASSERT(inputs.empty(), "Parameter cannot have inputs");
    // Note: ctor of shared_ptr isn't friend class for Expression -> we cannot use directly make_shared<Expression>(args)
    auto expr = std::shared_ptr<Expression>(new Expression(par, shape_infer_factory, false));
    create_expression_outputs(expr);
    expr->validate();
    return expr;
}

ExpressionPtr ExpressionFactory::create(const std::shared_ptr<ov::op::v0::Result>& res, const std::vector<PortConnectorPtr>& inputs,
                                        const std::shared_ptr<IShapeInferSnippetsFactory>& shape_infer_factory) {
    // Note: ctor of shared_ptr isn't friend class for Expression -> we cannot use directly make_shared<Expression>(args)
    auto expr = std::shared_ptr<Expression>(new Expression(res, shape_infer_factory));
    init_expression_inputs(expr, inputs);
    // The Result node don't need output port (because of sense of the node). But each node in openvino must have one output at least.
    // The port descriptors are automatically created in constructor. We manually clean output ports.
    expr->m_output_port_descriptors.clear();
    expr->validate();
    return expr;
}

ExpressionPtr ExpressionFactory::create(const std::shared_ptr<op::LoopBegin>& n, const std::vector<PortConnectorPtr>& inputs,
                                        const std::shared_ptr<IShapeInferSnippetsFactory>& shape_infer_factory) {
    OPENVINO_ASSERT(inputs.empty(), "LoopBegin cannot have inputs");
    auto expr = std::shared_ptr<Expression>(new Expression(n, shape_infer_factory, false));
    init_expression_inputs(expr, inputs);
    create_expression_outputs(expr);
    expr->validate();
    return expr;
}

ExpressionPtr ExpressionFactory::create(const std::shared_ptr<op::LoopEnd>& n, const std::vector<PortConnectorPtr>& inputs,
                                        const std::shared_ptr<IShapeInferSnippetsFactory>& shape_infer_factory) {
    auto expr = std::shared_ptr<Expression>(new Expression(n, shape_infer_factory, false));
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

ExpressionPtr ExpressionFactory::create(const std::shared_ptr<op::RegSpillBegin>& n, const std::vector<PortConnectorPtr>& inputs,
                                        const std::shared_ptr<IShapeInferSnippetsFactory>& shape_infer_factory) {
    auto expr = std::shared_ptr<Expression>(new Expression(n, shape_infer_factory, false));
    OPENVINO_ASSERT(inputs.empty(), "RegSpillBegin expression expects no inputs");
    const auto num_to_spill = n->get_regs_to_spill().size();
    expr->m_output_port_descriptors.resize(num_to_spill, nullptr);
    for (size_t i = 0; i < num_to_spill; i++)
        expr->m_output_port_descriptors[i] = std::make_shared<PortDescriptor>();
    expr->m_output_port_connectors.resize(num_to_spill, nullptr);
    for (size_t i = 0; i < num_to_spill; i++) {
        const auto source = expr->get_output_port(i);
        expr->m_output_port_connectors[i] = std::make_shared<PortConnector>(source);
    }
    expr->validate();
    return expr;
}

ExpressionPtr ExpressionFactory::create(const std::shared_ptr<op::RegSpillEnd>& n, const std::vector<PortConnectorPtr>& inputs,
                                        const std::shared_ptr<IShapeInferSnippetsFactory>& shape_infer_factory) {
    auto expr = std::shared_ptr<Expression>(new Expression(n, shape_infer_factory, false));
    const auto spill_begin_node = n->get_reg_spill_begin();
    const auto num_to_spill = spill_begin_node->get_regs_to_spill().size();
    OPENVINO_ASSERT(inputs.size() == num_to_spill, "Invalid num inputs for RegSpillEnd expression");
    expr->m_input_port_descriptors.resize(num_to_spill, std::make_shared<PortDescriptor>());
    init_expression_inputs(expr, inputs);
    expr->m_output_port_descriptors.clear();
    expr->validate();
    return expr;
}

#ifdef SNIPPETS_DEBUG_CAPS
ExpressionPtr ExpressionFactory::create(const std::shared_ptr<op::PerfCountBeginBase>& n, const std::vector<PortConnectorPtr>& inputs,
                                        const std::shared_ptr<IShapeInferSnippetsFactory>& shape_infer_factory) {
    OPENVINO_ASSERT(inputs.empty(), "PerfCountBegin shape_infer_factory do not accept any input connectors");
    return create_without_connections(n, shape_infer_factory);
}

ExpressionPtr ExpressionFactory::create(const std::shared_ptr<op::PerfCountEndBase>& n,
                                        const std::vector<PortConnectorPtr>& inputs,
                                        const std::shared_ptr<IShapeInferSnippetsFactory>& shape_infer_factory) {
    OPENVINO_ASSERT(inputs.empty(), "PerfCountEnd shape_infer_factory do not accept any input connectors");
    return create_without_connections(n, shape_infer_factory);
}

ExpressionPtr ExpressionFactory::create_without_connections(const std::shared_ptr<ov::Node>& n,
                                                            const std::shared_ptr<IShapeInferSnippetsFactory>& shape_infer_factory) {
    auto expr = std::shared_ptr<Expression>(new Expression(n, shape_infer_factory, false));
    expr->m_input_port_descriptors.clear();
    expr->m_output_port_descriptors.clear();
    expr->validate();
    return expr;
}
#endif

}// namespace lowered
}// namespace snippets
}// namespace ov
