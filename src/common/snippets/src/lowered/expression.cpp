// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/expression.hpp"

#include <snippets/itt.hpp>
#include "snippets/utils.hpp"

#include <openvino/core/graph_util.hpp>
#include <openvino/core/type.hpp>

namespace ngraph {
namespace snippets {
namespace lowered {

size_t Expression::LOOP_NULL_ID = SIZE_MAX;

ExpressionPort::ExpressionPort(const ExpressionPtr& expr, size_t port, Type type) : expr(expr), port(port), m_type(type) {
    if (type == Type::Input) {
        OPENVINO_ASSERT(port < expr->get_inputs().size(), "The input port must be less than input count");
    } else if (type == Type::Output) {
        OPENVINO_ASSERT(port < expr->get_outputs().size(), "The output port must be less than output count");
    }
}

Expression::Expression(const std::shared_ptr<Node>& n)
    : m_source_node{n}, m_emitter{nullptr}, m_reg_info{{}, {}}, m_is_outside_loop(utils::get_outside_loop_value(n)) {
    for (const auto& in : n->inputs())
        m_inputs.emplace_back(get_tensor_descriptor_ptr(in.get_source_output()));
    for (const auto& out : n->outputs())
        m_outputs.emplace_back(get_tensor_descriptor_ptr(out));
}

Expression::Expression(const std::shared_ptr<Node>& n, std::vector<TensorDescriptorPtr> inputs)
    : m_source_node{n}, m_emitter{nullptr}, m_inputs(std::move(inputs)), m_reg_info{{}, {}}, m_is_outside_loop(utils::get_outside_loop_value(n)) {
    for (const auto& out : n->outputs())
        m_outputs.emplace_back(get_tensor_descriptor_ptr(out));
}

Expression::Expression(const std::shared_ptr<Node>& n, std::vector<TensorDescriptorPtr> inputs, std::vector<TensorDescriptorPtr> outputs)
        : m_source_node{n}, m_emitter{nullptr}, m_inputs(std::move(inputs)), m_outputs(std::move(outputs)),
          m_reg_info{{}, {}}, m_is_outside_loop(utils::get_outside_loop_value(n)) {}

std::shared_ptr<Node> Expression::get_node() const {
    if (!m_source_node)
        throw ngraph_error("An attempt to get uninitialized node from lowered expression");
    return  m_source_node;
}

std::shared_ptr<Emitter> Expression::get_emitter() const {
    return  m_emitter;
}

void Expression::init_emitter(const std::shared_ptr<const TargetMachine>& target) {
    m_emitter = target->get(m_source_node->get_type_info())(m_source_node);
}

void Expression::replace_input(size_t port, TensorDescriptorPtr to) {
    OPENVINO_ASSERT(port < m_inputs.size(), "Failed to replace: target input port must be less than input count!");
    m_inputs[port] = std::move(to);
}

void Expression::replace_output(size_t port, TensorDescriptorPtr to) {
    OPENVINO_ASSERT(port < m_outputs.size(), "Failed to replace: target output port must be less than output count!");
    m_outputs[port] = std::move(to);
}

void Expression::set_loop_id(size_t id, size_t idx) {
    OPENVINO_ASSERT((std::find(m_loop_ids.begin(), m_loop_ids.end(), id) == m_loop_ids.end()),
                    "Expression cannot have several the same Loops");
    if (m_loop_ids.size() <= idx) {
        m_loop_ids.resize(idx + 1, LOOP_NULL_ID);
    }
    m_loop_ids[idx] = id;
}

void Expression::remove_loop_id(size_t id) {
    auto it = std::find(m_loop_ids.begin(), m_loop_ids.end(), id);
    OPENVINO_ASSERT(it == m_loop_ids.end(), "Expression doesn't have the Loop with ID " + std::to_string(id));
    *it = Expression::LOOP_NULL_ID;
}

ExpressionPort Expression::input_port(size_t i) {
    OPENVINO_ASSERT(i < m_inputs.size(), "Failed to get input port: target input port must be less than input count!");
    return ExpressionPort(this->shared_from_this(), i, ExpressionPort::Type::Input);
}

ExpressionPort Expression::output_port(size_t i) {
    OPENVINO_ASSERT(i < m_outputs.size(), "Failed to get output port: target output port must be less than output count!");
    return ExpressionPort(this->shared_from_this(), i, ExpressionPort::Type::Output);
}

IOExpression::IOExpression(const std::shared_ptr<ov::opset1::Parameter>& par, int64_t index)
        : Expression(par), m_index(index), m_type{io_type::INPUT} {
}

IOExpression::IOExpression(const std::shared_ptr<ov::opset1::Result>& res, int64_t index, std::vector<TensorDescriptorPtr> inputs)
        : Expression(res, inputs, {}), m_index(index), m_type{io_type::OUTPUT} {
}

bool operator==(const ExpressionPort& lhs, const ExpressionPort& rhs) {
    if (&lhs == &rhs)
        return true;
    OPENVINO_ASSERT(lhs.get_type() == rhs.get_type(), "Incorrect comparison: Ports are from different types!");
    return lhs.expr == rhs.expr && lhs.port == rhs.port;
}

bool operator!=(const ExpressionPort& lhs, const ExpressionPort& rhs) {
    return !(lhs == rhs);
}

bool operator<(const ExpressionPort& lhs, const ExpressionPort& rhs) {
    OPENVINO_ASSERT(lhs.get_type() == rhs.get_type(), "Incorrect comparison: Ports are from different types!");
    // Firstly ports
    return (lhs.port < rhs.port) || (lhs.port == rhs.port && lhs.expr < rhs.expr);
}
}// namespace lowered
}// namespace snippets
}// namespace ngraph
