// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/loop_port.hpp"

#include "snippets/utils/utils.hpp"


namespace ov {
namespace snippets {
namespace lowered {

LoopPort::LoopPort(const ExpressionPort& port, bool is_incremented, size_t dim_idx)
    : m_expr_port(std::make_shared<ExpressionPort>(port)), m_is_incremented(is_incremented) {
    set_dim_idx(dim_idx);
}

std::shared_ptr<LoopPort> LoopPort::clone_with_new_expr(const ExpressionPtr& new_expr) const {
    auto new_loop_port = std::make_shared<LoopPort>(*this);
    new_loop_port->m_expr_port = m_expr_port->clone_with_new_expr(new_expr);
    return new_loop_port;
}

void LoopPort::set_expr_port(std::shared_ptr<ExpressionPort> p) {
    OPENVINO_ASSERT(p, "Expression port is missed");
    m_expr_port = std::move(p);
}

void LoopPort::set_is_incremented(bool is_inc) {
    m_is_incremented = is_inc;
}

void LoopPort::set_dim_idx(size_t idx) {
    OPENVINO_ASSERT(idx < m_expr_port->get_descriptor_ptr()->get_shape().size(),
                    "LoopPort dim_idx (",
                    idx,
                    ") must be less than the corresponding expression port shape rank (",
                    m_expr_port->get_descriptor_ptr()->get_shape().size(),
                    ")");
    m_dim_idx = idx;
}

bool operator==(const LoopPort& lhs, const LoopPort& rhs) {
    if (&lhs == &rhs)
        return true;
    return *lhs.m_expr_port == *rhs.m_expr_port && lhs.m_is_incremented == rhs.m_is_incremented && lhs.m_dim_idx == rhs.m_dim_idx;
}

bool operator!=(const LoopPort& lhs, const LoopPort& rhs) {
    return !(lhs == rhs);
}

bool operator<(const LoopPort& lhs, const LoopPort& rhs) {
    return (*lhs.m_expr_port < *rhs.m_expr_port) ||
           (*lhs.m_expr_port == *rhs.m_expr_port &&
            (lhs.m_is_incremented < rhs.m_is_incremented ||
             (lhs.m_is_incremented == rhs.m_is_incremented && lhs.m_dim_idx < rhs.m_dim_idx)));
}

} // namespace lowered
} // namespace snippets
} // namespace ov
