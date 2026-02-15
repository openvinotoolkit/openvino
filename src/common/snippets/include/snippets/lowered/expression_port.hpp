// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>

#include "port_descriptor.hpp"

namespace ov::snippets::lowered {

class PortConnector;
class Expression;
class ExpressionPort {
public:
    enum Type : uint8_t { Input, Output };

    ExpressionPort() = default;
    explicit ExpressionPort(const std::shared_ptr<Expression>& expr, Type type, size_t port);
    /**
     * @interface clone_with_new_expr
     * @brief Creates similar Expression port, but for new expression
     */
    [[nodiscard]] std::shared_ptr<ExpressionPort> clone_with_new_expr(
        const std::shared_ptr<Expression>& new_expr) const;

    [[nodiscard]] std::shared_ptr<Expression> get_expr() const;
    [[nodiscard]] Type get_type() const {
        return m_type;
    }
    [[nodiscard]] size_t get_index() const {
        return m_port_index;
    }

    [[nodiscard]] const PortDescriptorPtr& get_descriptor_ptr() const;
    [[nodiscard]] const std::shared_ptr<PortConnector>& get_port_connector_ptr() const;
    // Returns connected ports to the current:
    //  - Input port returns one source (parent) port
    //  - Output port returns all consumer ports (children)
    [[nodiscard]] std::set<ExpressionPort> get_connected_ports() const;

    // Note: It may be called only for input expression ports
    //       since output ports don't support PortConnector changing (this is determined by the creation of the
    //       expression)
    void replace_input_port_connector(std::shared_ptr<PortConnector> to) const;

    friend bool operator==(const ExpressionPort& lhs, const ExpressionPort& rhs);
    friend bool operator!=(const ExpressionPort& lhs, const ExpressionPort& rhs);
    friend bool operator<(const ExpressionPort& lhs, const ExpressionPort& rhs);

private:
    std::weak_ptr<Expression> m_expr;
    Type m_type = Type::Output;
    size_t m_port_index = 0;
};

void replace_input_port_connectors(const std::set<ExpressionPort>& consumers, const std::shared_ptr<PortConnector>& to);

}  // namespace ov::snippets::lowered
