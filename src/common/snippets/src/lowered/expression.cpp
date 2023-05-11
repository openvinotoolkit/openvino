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

Expression::Expression(const std::shared_ptr<Node>& n)
    : m_source_node{n}, m_emitter{nullptr}, m_input_tensors{}, m_output_tensors{}, m_reg_info{{}, {}} {
    m_input_port_descriptors.reserve(n->get_input_size());
    m_output_port_descriptors.reserve(n->get_output_size());
    for (const auto& input : n->inputs()) {
        m_input_port_descriptors.push_back(PortManager::get_port_descriptor_ptr(input));
    }
    for (const auto& output : n->outputs()) {
        m_output_port_descriptors.push_back(PortManager::get_port_descriptor_ptr(output));
    }
}

const TensorPtr& Expression::get_input_tensor(size_t i) const {
    OPENVINO_ASSERT(i < m_input_tensors.size(), "Failed to get input tensor: target input port must be less than input count!");
    return m_input_tensors[i];
}
const TensorPtr& Expression::get_output_tensor(size_t i) const {
    OPENVINO_ASSERT(i < m_output_tensors.size(), "Failed to get output: target output port must be less than output count!");
    return m_output_tensors[i];
}

const PortDescriptorPtr& Expression::get_input_port_descriptor(size_t i) const {
    OPENVINO_ASSERT(i < m_input_port_descriptors.size(), "Failed to get input port descriptor: target input port must be less than input count!");
    return m_input_port_descriptors[i];
}
const PortDescriptorPtr& Expression::get_output_port_descriptor(size_t i) const {
    OPENVINO_ASSERT(i < m_output_port_descriptors.size(), "Failed to get output port descriptor: target output port must be less than output count!");
    return m_output_port_descriptors[i];
}

std::shared_ptr<Node> Expression::get_node() const {
    if (!m_source_node)
        OPENVINO_THROW("An attempt to get uninitialized node from lowered expression");
    return  m_source_node;
}

std::shared_ptr<Emitter> Expression::get_emitter() const {
    return m_emitter;
}

void Expression::init_emitter(const std::shared_ptr<const TargetMachine>& target) {
    m_emitter = target->get(m_source_node->get_type_info())(m_source_node);
}

void Expression::validate() const {
    OPENVINO_ASSERT(m_input_port_descriptors.size() == m_input_tensors.size(), "The count of input ports and input tensors must be equal");
    OPENVINO_ASSERT(m_output_port_descriptors.size() == m_output_tensors.size(), "The count of output ports and output tensors must be equal");
    OPENVINO_ASSERT(m_source_node != nullptr, "The expression has null source node");
}

void Expression::replace_input(size_t port, TensorPtr to) {
    OPENVINO_ASSERT(port < m_input_tensors.size(), "Failed to replace: target input port must be less than input count!");
    m_input_tensors[port] = std::move(to);
}

void Expression::set_loop_id(size_t id, size_t idx) {
    if (id != LOOP_NULL_ID) {
        OPENVINO_ASSERT((std::find(m_loop_ids.begin(), m_loop_ids.end(), id) == m_loop_ids.end()),
                        "Expression cannot have several the same Loops");
    }
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

ExpressionPort Expression::get_input_port(size_t i) {
    return ExpressionPort(this->shared_from_this(), ExpressionPort::Type::Input, i);
}

ExpressionPort Expression::get_output_port(size_t i) {
    return ExpressionPort(this->shared_from_this(), ExpressionPort::Type::Output, i);
}

IOExpression::IOExpression(const std::shared_ptr<ov::opset1::Parameter>& par, int64_t index)
        : Expression(par), m_index(index), m_type{io_type::INPUT} {}
IOExpression::IOExpression(const std::shared_ptr<ov::opset1::Result>& res, int64_t index)
        : Expression(res), m_index(index), m_type{io_type::OUTPUT} {}

}// namespace lowered
}// namespace snippets
}// namespace ngraph
