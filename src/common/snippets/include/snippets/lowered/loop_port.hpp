// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/expression_port.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/utils/utils.hpp"


namespace ov {
namespace snippets {
namespace lowered {

/* The class describes port of Loop: expression port that connected to Expressions from other Loops.
 */
class LoopPort {
public:
    enum {UNDEFINED_DIM_IDX = std::numeric_limits<size_t>::max()};
    enum class Type {
        Incremented,    // Loop port which data ptr should be incremented after each Loop iteration
        NotIncremented, // Loop port which data ptr should not be to avoid double increment
        NotProcessed,   // LoopPort which doesn't process the dim by `dim_idx` (UNDEFINED_DIM_IDX) and is used only for Loop bound definition
    };

    LoopPort() = default;

    template<LoopPort::Type T = Type::Incremented,
            typename std::enable_if<T == Type::Incremented || T == Type::NotIncremented, bool>::type = true>
    static LoopPort create(const ExpressionPort& port, size_t dim_idx = 0) {
        return LoopPort(port, dim_idx, T);
    }

    template<LoopPort::Type T,
            typename std::enable_if<T == Type::NotProcessed, bool>::type = true>
    static LoopPort create(const ExpressionPort& port) {
        return LoopPort(port, UNDEFINED_DIM_IDX, Type::NotProcessed);
    }

    std::shared_ptr<LoopPort> clone_with_new_expr(const ExpressionPtr& new_expr) const;

    friend bool operator==(const LoopPort& lhs, const LoopPort& rhs);
    friend bool operator!=(const LoopPort& lhs, const LoopPort& rhs);
    friend bool operator<(const LoopPort& lhs, const LoopPort& rhs);

    const std::shared_ptr<ExpressionPort>& get_expr_port() const { return m_expr_port; }
    Type get_type() const { return m_type; }
    size_t get_dim_idx() const { return m_dim_idx; }

    void set_expr_port(std::shared_ptr<ExpressionPort> p);
    void set_type(Type type);
    void set_dim_idx(size_t idx);

private:
    LoopPort(const ExpressionPort& port, size_t dim_idx, Type type);

    std::shared_ptr<ExpressionPort> m_expr_port = {};
    size_t m_dim_idx = 0; // The numeration starts from the end (dim_idx = 0 -> is the most inner dimension)
    Type m_type = Type::Incremented;
};

inline std::ostream& operator<<(std::ostream& out, const LoopPort::Type& type) {
    switch (type) {
    case LoopPort::Type::Incremented:
        out << "Incremented";
        break;
    case LoopPort::Type::NotIncremented:
        out << "NotIncremented";
        break;
    case LoopPort::Type::NotProcessed:
        out << "NotProcessed";
        break;
    default:
        OPENVINO_THROW("Unknown LoopPort Type");
    }
    return out;
}

} // namespace lowered
} // namespace snippets
} // namespace ov
