// Copyright (C) 2018-2025 Intel Corporation
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
        NotIncremented, // Loop port which data ptr should not be incremented (for example, to avoid double increment)
        NotProcessed,   // LoopPort which doesn't process the dim by `dim_idx` (UNDEFINED_DIM_IDX) and is used only for Loop bound definition
    };

    LoopPort() = default;

    template<LoopPort::Type T,
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
    size_t get_dim_idx() const;

    void set_expr_port(std::shared_ptr<ExpressionPort> p);
    void set_dim_idx(size_t idx);

    template<LoopPort::Type T,
             typename std::enable_if<T == Type::Incremented || T == Type::NotIncremented, bool>::type = true>
    void convert_to_type() {
        OPENVINO_ASSERT(is_processed(), "NotProcessed LoopPort cannot change type!");
        m_type = T;
    }

    bool is_processed() const;
    bool is_incremented() const;

private:
    LoopPort(const ExpressionPort& port, size_t dim_idx, Type type);

    std::shared_ptr<ExpressionPort> m_expr_port = {};
    size_t m_dim_idx = 0; // The numeration starts from the end (dim_idx = 0 -> is the most inner dimension)
    Type m_type = Type::Incremented;
};

std::ostream& operator<<(std::ostream& out, const LoopPort::Type& type);

} // namespace lowered
} // namespace snippets
} // namespace ov
