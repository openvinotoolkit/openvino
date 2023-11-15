// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "port_descriptor.hpp"


namespace ov {
namespace snippets {
namespace lowered {

class PortConnector;
class Expression;
class ExpressionPort {
public:
    enum Type {
        Input,
        Output
    };

    ExpressionPort() = default;
    explicit ExpressionPort(const std::shared_ptr<Expression>& expr, Type type, size_t port);
    /**
    * @interface clone_with_new_expr
    * @brief Creates similar Expression port, but for new expression
     */
    std::shared_ptr<ExpressionPort> clone_with_new_expr(const std::shared_ptr<Expression>& new_expr) const;

    std::shared_ptr<Expression> get_expr() const;
    Type get_type() const { return m_type; }
    size_t get_index() const { return m_port_index; }

    const PortDescriptorPtr& get_descriptor_ptr() const;
    const std::shared_ptr<PortConnector>& get_port_connector_ptr() const;
    // Returns connected ports to the current:
    //  - Input port returns one source (parent) port
    //  - Output port returns all consumer ports (children)
    std::set<ExpressionPort> get_connected_ports() const;

    friend bool operator==(const ExpressionPort& lhs, const ExpressionPort& rhs);
    friend bool operator!=(const ExpressionPort& lhs, const ExpressionPort& rhs);
    friend bool operator<(const ExpressionPort& lhs, const ExpressionPort& rhs);

private:
    std::weak_ptr<Expression> m_expr;
    Type m_type = Type::Output;
    size_t m_port_index = 0;
};
} // namespace lowered
} // namespace snippets
} // namespace ov
