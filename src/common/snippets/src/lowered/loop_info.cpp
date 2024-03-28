// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/loop_info.hpp"

#include "snippets/utils.hpp"


namespace ov {
namespace snippets {
namespace lowered {

LoopInfo::LoopInfo(size_t work_amount,
                   size_t increment,
                   const std::vector<LoopPort>& entries,
                   const std::vector<LoopPort>& exits,
                   const SpecificIterationHandlers& handlers)
    : m_work_amount(work_amount),
      m_increment(increment),
      m_entry_points(entries),
      m_exit_points(exits),
      m_handlers(handlers) {}

LoopInfo::LoopInfo(size_t work_amount,
                   size_t increment,
                   const std::vector<ExpressionPort>& entries,
                   const std::vector<ExpressionPort>& exits,
                   const SpecificIterationHandlers& handlers)
    : m_work_amount(work_amount),
      m_increment(increment),
      m_handlers(handlers) {
    m_entry_points.reserve(entries.size());
    m_exit_points.reserve(exits.size());
    for (const auto& port : entries)
        m_entry_points.emplace_back(port);
    for (const auto& port : exits)
        m_exit_points.emplace_back(port);
}

std::shared_ptr<LoopInfo> LoopInfo::clone_with_new_expr(const ExpressionMap& expr_map) const {
    auto clone_loop_ports = [&expr_map](const std::vector<LoopPort>& port_points) {
        std::vector<LoopPort> cloned_port_points;
        cloned_port_points.reserve(port_points.size());
        for (const auto& p : port_points) {
            const auto& expr = p.expr_port->get_expr().get();
            OPENVINO_ASSERT(expr_map.count(expr), "Can't clone LoopInfo: old expression is not in the map");
            const auto& new_expr = expr_map.at(expr);
            cloned_port_points.emplace_back(*p.clone_with_new_expr(new_expr));
        }
        return cloned_port_points;
    };
    const auto& new_entry_points = clone_loop_ports(m_entry_points);
    const auto& new_exit_points = clone_loop_ports(m_exit_points);

    return std::make_shared<LoopInfo>(m_work_amount, m_increment, new_entry_points, new_exit_points, m_handlers);
}

size_t LoopInfo::get_work_amount() const {
    return m_work_amount;
}

size_t LoopInfo::get_increment() const {
    return m_increment;
}

const std::vector<LoopPort>& LoopInfo::get_entry_points() const {
    return m_entry_points;
}

const std::vector<LoopPort>& LoopInfo::get_exit_points() const {
    return m_exit_points;
}

const SpecificIterationHandlers& LoopInfo::get_handlers() const {
    return m_handlers;
}

size_t LoopInfo::get_dim_idx() const {
    OPENVINO_ASSERT(!m_entry_points.empty(), "Loop info must have at least one entry point");
    auto equal_dim_idxes = [&](const LoopPort& p) {
        return p.dim_idx == m_entry_points[0].dim_idx;
    };
    if (std::all_of(m_entry_points.begin(), m_entry_points.end(), equal_dim_idxes) &&
        std::all_of(m_exit_points.begin(), m_exit_points.end(), equal_dim_idxes)) {
        return m_entry_points[0].dim_idx;
    } else {
        return UNDEFINED_DIM_IDX;
    }
}

void LoopInfo::set_dim_idx(size_t dim_idx) {
    auto set_common_dim_idx = [dim_idx](LoopPort& port) {
        port.dim_idx = dim_idx;
    };
    update_entry_points(set_common_dim_idx);
    update_exit_points(set_common_dim_idx);
}

void LoopInfo::set_work_amount(size_t work_amount) {
    m_work_amount = work_amount;
}

void LoopInfo::set_increment(size_t increment) {
    m_increment = increment;
}

void LoopInfo::set_entry_points(std::vector<LoopPort> entry_points) {
    m_entry_points = std::move(entry_points);
}

void LoopInfo::set_exit_points(std::vector<LoopPort> exit_points) {
    m_exit_points = std::move(exit_points);
}

void LoopInfo::set_handlers(SpecificIterationHandlers handlers) {
    m_handlers = std::move(handlers);
}

void LoopInfo::update_entry_points(const std::function<void(LoopPort&)>& updater) {
    std::for_each(m_entry_points.begin(), m_entry_points.end(), updater);
}

void LoopInfo::update_exit_points(const std::function<void(LoopPort&)>& updater) {
    std::for_each(m_exit_points.begin(), m_exit_points.end(), updater);
}

} // namespace lowered
} // namespace snippets
} // namespace ov
