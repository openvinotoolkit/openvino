// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/loop_info.hpp"

#include "snippets/utils.hpp"


namespace ov {
namespace snippets {
namespace lowered {

LoopInfo::LoopInfo(size_t work_amount, size_t increment, const std::vector<LoopPort>& entries, const std::vector<LoopPort>& exits)
    : m_work_amount(work_amount), m_increment(increment), m_entry_points(entries), m_exit_points(exits) {}

LoopInfo::LoopInfo(size_t work_amount, size_t increment, const std::vector<ExpressionPort>& entries, const std::vector<ExpressionPort>& exits)
    : m_work_amount(work_amount), m_increment(increment) {
    m_entry_points.reserve(entries.size());
    m_exit_points.reserve(exits.size());
    for (const auto& port : entries)
        m_entry_points.emplace_back(port);
    for (const auto& port : exits)
        m_exit_points.emplace_back(port);
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

void LoopInfo::set_work_amount(size_t work_amount) {
    m_work_amount = work_amount;
}

void LoopInfo::set_increment(size_t increment) {
    m_increment = increment;
}

void LoopInfo::set_dim_idx(size_t dim_idx) {
    auto set_common_dim_idx = [dim_idx](LoopPort& port) {
        port.dim_idx = dim_idx;
    };
    update_entry_points(set_common_dim_idx);
    update_exit_points(set_common_dim_idx);
}

void LoopInfo::set_entry_points(std::vector<LoopPort> entry_points) {
    m_entry_points = std::move(entry_points);
}

void LoopInfo::set_exit_points(std::vector<LoopPort> exit_points) {
    m_exit_points = std::move(exit_points);
}

std::vector<bool> LoopInfo::get_is_incremented() const {
    std::vector<bool> values;
    values.reserve(m_entry_points.size() + m_exit_points.size());
    auto initializer = [&values](const LoopPort& port) {
        values.push_back(port.is_incremented);
    };
    init_using_entry_points(initializer);
    init_using_exit_points(initializer);
    return values;
}

std::vector<int64_t> LoopInfo::get_ptr_increments() const {
    std::vector<int64_t> values;
    values.reserve(m_entry_points.size() + m_exit_points.size());
    auto initializer = [&values](const LoopPort& port) {
        values.push_back(port.ptr_increment);
    };
    init_using_entry_points(initializer);
    init_using_exit_points(initializer);
    return values;
}

std::vector<int64_t> LoopInfo::get_finalization_offsets() const {
    std::vector<int64_t> values;
    values.reserve(m_entry_points.size() + m_exit_points.size());
    auto initializer = [&values](const LoopPort& port) {
        values.push_back(port.finalization_offset);
    };
    init_using_entry_points(initializer);
    init_using_exit_points(initializer);
    return values;
}

std::vector<int64_t> LoopInfo::get_data_sizes() const {
    std::vector<int64_t> values;
    values.reserve(m_entry_points.size() + m_exit_points.size());
    auto initializer = [&values](const LoopPort& port) {
        values.push_back(port.data_size);
    };
    init_using_entry_points(initializer);
    init_using_exit_points(initializer);
    return values;
}

std::vector<LoopPort> LoopInfo::clone_loop_ports(const ExpressionMap& expr_map, const std::vector<LoopPort>& port_ports) {
    std::vector<LoopPort> cloned_port_points;
    cloned_port_points.reserve(port_ports.size());
    for (const auto& p : port_ports) {
        const auto& expr = p.expr_port->get_expr().get();
        OPENVINO_ASSERT(expr_map.count(expr), "Can't clone LoopInfo: old expression is not in the map");
        const auto& new_expr = expr_map.at(expr);
        cloned_port_points.emplace_back(*p.clone_with_new_expr(new_expr));
    }
    return cloned_port_points;
}

UnifiedLoopInfo::UnifiedLoopInfo(size_t work_amount,
                             size_t increment,
                             const std::vector<LoopPort>& entries,
                             const std::vector<LoopPort>& exits,
                             const SpecificIterationHandlers& handlers)
    : LoopInfo(work_amount, increment, entries, exits), m_handlers(handlers) {}

UnifiedLoopInfo::UnifiedLoopInfo(size_t work_amount,
                             size_t increment,
                             const std::vector<ExpressionPort>& entries,
                             const std::vector<ExpressionPort>& exits,
                             const SpecificIterationHandlers& handlers)
    : LoopInfo(work_amount, increment, entries, exits), m_handlers(handlers) {}

std::shared_ptr<LoopInfo> UnifiedLoopInfo::clone_with_new_expr(const ExpressionMap& expr_map) const {
    const auto& new_entry_points = clone_loop_ports(expr_map, m_entry_points);
    const auto& new_exit_points = clone_loop_ports(expr_map, m_exit_points);

    return std::make_shared<UnifiedLoopInfo>(m_work_amount, m_increment, new_entry_points, new_exit_points, m_handlers);
}

const SpecificIterationHandlers& UnifiedLoopInfo::get_handlers() const {
    return m_handlers;
}

void UnifiedLoopInfo::set_handlers(SpecificIterationHandlers handlers) {
    m_handlers = std::move(handlers);
}

ExpandedLoopInfo::ExpandedLoopInfo(size_t work_amount, size_t increment,
                                   const std::vector<LoopPort>& entries, const std::vector<LoopPort>& exits,
                                   SpecificLoopIterType type, std::shared_ptr<UnifiedLoopInfo> unified_loop_info)
    : LoopInfo(work_amount, increment, entries, exits), m_type(type) {
    OPENVINO_ASSERT(unified_loop_info, "Failed to create ExpandedLoopInfo: unified loop info is nullptr!");
    m_unified_loop_info = std::move(unified_loop_info);

    m_ptr_increments = get_ptr_increments();
    m_finalization_offsets = get_finalization_offsets();
    m_data_sizes = get_data_sizes();
}

ExpandedLoopInfo::ExpandedLoopInfo(size_t work_amount, size_t increment,
                                   const std::vector<LoopPort>& entries, const std::vector<LoopPort>& exits,
                                   std::vector<int64_t> ptr_increments, std::vector<int64_t> final_offsets, std::vector<int64_t> data_sizes,
                                   SpecificLoopIterType type, std::shared_ptr<UnifiedLoopInfo> unified_loop_info)
    : LoopInfo(work_amount, increment, entries, exits), m_ptr_increments(std::move(ptr_increments)), m_finalization_offsets(std::move(final_offsets)),
      m_data_sizes(std::move(data_sizes)), m_type(type) {
    OPENVINO_ASSERT(unified_loop_info, "Failed to create ExpandedLoopInfo: unified loop info is nullptr!");
    m_unified_loop_info = std::move(unified_loop_info);

    const auto count = entries.size() + exits.size();
    OPENVINO_ASSERT(utils::everyone_is(count, m_ptr_increments.size(), m_finalization_offsets.size(), m_data_sizes.size()),
                    "Incompatible data ptr shifts!");
}

std::shared_ptr<LoopInfo> ExpandedLoopInfo::clone_with_new_expr(const ExpressionMap& expr_map) const {
    const auto& new_entry_points = clone_loop_ports(expr_map, m_entry_points);
    const auto& new_exit_points = clone_loop_ports(expr_map, m_exit_points);

    return std::make_shared<ExpandedLoopInfo>(m_work_amount, m_increment, new_entry_points, new_exit_points,
                                              m_ptr_increments, m_finalization_offsets, m_data_sizes, m_type, m_unified_loop_info);
}

const std::shared_ptr<UnifiedLoopInfo>& ExpandedLoopInfo::get_unified_loop_info() const {
    OPENVINO_ASSERT(m_unified_loop_info, "Failed to get unified loop info: it's nullptr");
    return m_unified_loop_info;
}

SpecificLoopIterType ExpandedLoopInfo::get_type() const {
    return m_type;
}

const pass::PassPipeline& ExpandedLoopInfo::get_handlers_by_type() const {
    return get_unified_loop_info()->get_handlers().get_handlers_by_type(m_type);
}

std::vector<int64_t>& ExpandedLoopInfo::get_dense_ptr_increments() {
    return m_ptr_increments;
}

std::vector<int64_t>& ExpandedLoopInfo::get_dense_finalization_offsets() {
    return m_finalization_offsets;
}

const std::vector<int64_t>& ExpandedLoopInfo::get_dense_data_sizes() const {
    return m_data_sizes;
}

} // namespace lowered
} // namespace snippets
} // namespace ov
