// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/expression_port.hpp"

#include "snippets/utils/utils.hpp"


namespace ov {
namespace snippets {
namespace lowered {

ExpressionPort::ExpressionPort(const std::shared_ptr<Expression>& expr, Type type, size_t port)
        : m_expr(expr), m_type(type), m_port_index(port) {}

std::shared_ptr<ExpressionPort> ExpressionPort::clone_with_new_expr(const std::shared_ptr<Expression>& new_expr) const {
    return std::make_shared<ExpressionPort>(new_expr, m_type, m_port_index);
}

std::shared_ptr<Expression> ExpressionPort::get_expr() const {
    const auto expr_ptr = m_expr.lock();
    OPENVINO_ASSERT(expr_ptr != nullptr, "ExpressionPort has invalid expression pointer");
    return expr_ptr;
}

const PortDescriptorPtr& ExpressionPort::get_descriptor_ptr() const {
    const auto& descs = m_type == Type::Input ? get_expr()->m_input_port_descriptors
                                              : get_expr()->m_output_port_descriptors;
    OPENVINO_ASSERT(m_port_index < descs.size(), "Incorrect index of port");
    return descs[m_port_index];
}

const std::shared_ptr<PortConnector>& ExpressionPort::get_port_connector_ptr() const {
    const auto& connectors = m_type == Type::Input ? get_expr()->m_input_port_connectors
                                                   : get_expr()->m_output_port_connectors;
    OPENVINO_ASSERT(m_port_index < connectors.size(), "Incorrect index of port");
    return connectors[m_port_index];
}

std::set<ExpressionPort> ExpressionPort::get_connected_ports() const {
    if (ExpressionPort::m_type == Type::Input) {
        return { get_expr()->m_input_port_connectors[m_port_index]->get_source() };
    }
    if (ExpressionPort::m_type == Type::Output) {
        return get_expr()->m_output_port_connectors[m_port_index]->get_consumers();
    }
    OPENVINO_THROW("ExpressionPort supports only Input and Output types");
}

void ExpressionPort::replace_input_port_connector(std::shared_ptr<PortConnector> to) const {
    OPENVINO_ASSERT(m_type == Type::Input, "Only Input Expression ports can change the corresponding PortConnector!");
    get_expr()->set_input_port_connector(m_port_index, std::move(to));
}

bool operator==(const ExpressionPort& lhs, const ExpressionPort& rhs) {
    if (&lhs == &rhs)
        return true;
    OPENVINO_ASSERT(lhs.get_type() == rhs.get_type(), "Incorrect ExpressionPort comparison");
    return lhs.get_index() == rhs.get_index() && lhs.get_expr() == rhs.get_expr();
}
bool operator!=(const ExpressionPort& lhs, const ExpressionPort& rhs) {
    return !(lhs == rhs);
}
bool operator<(const ExpressionPort& lhs, const ExpressionPort& rhs) {
    OPENVINO_ASSERT(lhs.get_type() == rhs.get_type(), "Incorrect ExpressionPort comparison");
    return (lhs.get_index() < rhs.get_index()) || (lhs.get_index() == rhs.get_index() && lhs.get_expr() < rhs.get_expr());
}

void replace_input_port_connectors(const std::set<ExpressionPort>& consumers, const std::shared_ptr<PortConnector>& to) {
    for (const auto& consumer_input : consumers) {
        consumer_input.replace_input_port_connector(to);
    }
}

}// namespace lowered
}// namespace snippets
}// namespace ov
