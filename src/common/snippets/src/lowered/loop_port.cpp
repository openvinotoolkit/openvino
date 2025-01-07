// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/loop_port.hpp"
#include "snippets/lowered/loop_info.hpp"

#include "snippets/utils/utils.hpp"


namespace ov {
namespace snippets {
namespace lowered {

LoopPort::LoopPort(const ExpressionPort& port, size_t dim_idx, Type type)
    : m_expr_port(std::make_shared<ExpressionPort>(port)), m_type(type)  {
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

void LoopPort::set_type(Type type) {
    m_type = type;
}

void LoopPort::set_dim_idx(size_t idx) {
    if (get_type() == LoopPort::Type::NotProcessed) {
        OPENVINO_ASSERT(idx == LoopInfo::UNDEFINED_DIM_IDX, "NotProcessed LoopPort cah have only UNDEFINED_DIM_IDX");
    } else {
        OPENVINO_ASSERT(idx < m_expr_port->get_descriptor_ptr()->get_shape().size(),
                        "LoopPort dim_idx (",
                        idx,
                        ") must be less than the corresponding expression port shape rank (",
                        m_expr_port->get_descriptor_ptr()->get_shape().size(),
                        ")");
    }
    m_dim_idx = idx;
}

bool operator==(const LoopPort& lhs, const LoopPort& rhs) {
    if (&lhs == &rhs)
        return true;
    return *lhs.m_expr_port == *rhs.m_expr_port && lhs.m_type == rhs.m_type && lhs.m_dim_idx == rhs.m_dim_idx;
}

bool operator!=(const LoopPort& lhs, const LoopPort& rhs) {
    return !(lhs == rhs);
}

bool operator<(const LoopPort& lhs, const LoopPort& rhs) {
    return (*lhs.m_expr_port < *rhs.m_expr_port) ||
           (*lhs.m_expr_port == *rhs.m_expr_port &&
            (lhs.m_type < rhs.m_type ||
             (lhs.m_type == rhs.m_type && lhs.m_dim_idx < rhs.m_dim_idx)));
}

} // namespace lowered
} // namespace snippets
} // namespace ov
