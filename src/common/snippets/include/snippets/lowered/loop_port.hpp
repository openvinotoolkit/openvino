// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/expression_port.hpp"
#include "snippets/lowered/expression.hpp"


namespace ov {
namespace snippets {
namespace lowered {

/* The structure describes port of Loop: expression port that connected to Expressions from other Loops.
 */
struct LoopPort {
    LoopPort() = default;
    LoopPort(const ExpressionPort& port, bool is_incremented = true, size_t dim_idx = 0);

    std::shared_ptr<LoopPort> clone_with_new_expr(const ExpressionPtr& new_expr) const;

    friend bool operator==(const LoopPort& lhs, const LoopPort& rhs);
    friend bool operator!=(const LoopPort& lhs, const LoopPort& rhs);
    friend bool operator<(const LoopPort& lhs, const LoopPort& rhs);

    const std::shared_ptr<ExpressionPort>& get_expr_port() const { return m_expr_port; }
    bool is_incremented() const { return m_is_incremented; }
    size_t get_dim_idx() const { return m_dim_idx; }

    void set_expr_port(std::shared_ptr<ExpressionPort> p);
    void set_is_incremented(bool is_inc);
    void set_dim_idx(size_t idx);

private:
    std::shared_ptr<ExpressionPort> m_expr_port = {};
    // True if after each Loop iteration the corresponding data pointer should be incremented.
    // Otherwise, the data pointer shift is skipped
    bool m_is_incremented = true;
    size_t m_dim_idx = 0; // The numeration starts from the end (dim_idx = 0 -> is the most inner dimension)
};

} // namespace lowered
} // namespace snippets
} // namespace ov
