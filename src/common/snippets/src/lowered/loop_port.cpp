// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/loop_port.hpp"

#include "snippets/utils/utils.hpp"


namespace ov {
namespace snippets {
namespace lowered {

LoopPort::LoopPort(const ExpressionPort& port, bool is_incremented, size_t dim_idx)
    : expr_port(std::make_shared<ExpressionPort>(port)), is_incremented(is_incremented), dim_idx(dim_idx) {
    OPENVINO_ASSERT(dim_idx < port.get_descriptor_ptr()->get_shape().size(),
                    "LoopPort dim_idx (",
                    dim_idx,
                    ") must be less than the corresponding expression port shape rank (",
                    port.get_descriptor_ptr()->get_shape().size(),
                    ")");
}

std::shared_ptr<LoopPort> LoopPort::clone_with_new_expr(const ExpressionPtr& new_expr) const {
    auto new_loop_port = std::make_shared<LoopPort>(*this);
    new_loop_port->expr_port = expr_port->clone_with_new_expr(new_expr);
    return new_loop_port;
}

bool operator==(const LoopPort& lhs, const LoopPort& rhs) {
    if (&lhs == &rhs)
        return true;
    return lhs.expr_port == rhs.expr_port && lhs.is_incremented == rhs.is_incremented && lhs.dim_idx == rhs.dim_idx;
}

bool operator!=(const LoopPort& lhs, const LoopPort& rhs) {
    return !(lhs == rhs);
}

bool operator<(const LoopPort& lhs, const LoopPort& rhs) {
    return (lhs.expr_port < rhs.expr_port) ||
           (lhs.expr_port == rhs.expr_port &&
            (lhs.is_incremented < rhs.is_incremented ||
             (lhs.is_incremented == rhs.is_incremented && lhs.dim_idx < rhs.dim_idx)));
}

} // namespace lowered
} // namespace snippets
} // namespace ov
