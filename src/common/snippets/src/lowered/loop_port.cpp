// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/loop_port.hpp"

#include "snippets/utils/utils.hpp"


namespace ov {
namespace snippets {
namespace lowered {

LoopPort::LoopPort(const ExpressionPort& port, size_t dim_idx, Type type)
    : m_expr_port(std::make_shared<ExpressionPort>(port)), m_type(type)  {
    if (is_processed()) {
        set_dim_idx(dim_idx);
    } else {
        OPENVINO_ASSERT(dim_idx == UNDEFINED_DIM_IDX, "NotProcessed LoopPort can have only UNDEFINED_DIM_IDX");
        m_dim_idx = dim_idx;
    }
}

std::shared_ptr<LoopPort> LoopPort::clone_with_new_expr(const ExpressionPtr& new_expr) const {
    auto new_loop_port = std::make_shared<LoopPort>(*this);
    new_loop_port->m_expr_port = m_expr_port->clone_with_new_expr(new_expr);
    return new_loop_port;
}

bool LoopPort::is_processed() const {
    switch (m_type) {
        case Type::Incremented:
        case Type::NotIncremented:
            return true;
        case Type::NotProcessed:
            return false;
        default:
            OPENVINO_THROW("Unknown LoopPort type");
    }
}

bool LoopPort::is_incremented() const {
    return m_type == Type::Incremented;
}

size_t LoopPort::get_dim_idx() const {
    OPENVINO_ASSERT(is_processed(), "NotProcessed LoopPort cannot call `get_dim_idx()`");
    return m_dim_idx;
}

void LoopPort::set_expr_port(std::shared_ptr<ExpressionPort> p) {
    OPENVINO_ASSERT(p, "Expression port is missed");
    m_expr_port = std::move(p);
}

void LoopPort::set_dim_idx(size_t idx) {
    OPENVINO_ASSERT(is_processed(), "NotProcessed LoopPort cannot call `get_dim_idx()`");
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

std::ostream& operator<<(std::ostream& out, const LoopPort::Type& type) {
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
