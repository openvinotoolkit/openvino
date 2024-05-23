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

    std::shared_ptr<ExpressionPort> expr_port = {};
    // True if after each Loop iteration the corresponding data pointer should be incremented.
    // Otherwise, the data pointer shift is skipped
    bool is_incremented = true;
    size_t dim_idx = 0; // The numeration starts from the end (dim_idx = 0 -> is the most inner dimension)
};

} // namespace lowered
} // namespace snippets
} // namespace ov
