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

size_t LoopInfo::get_entry_count() const {
    return m_entry_points.size();
}

size_t LoopInfo::get_exit_count() const {
    return m_exit_points.size();
}

size_t LoopInfo::get_work_amount() const {
    return m_work_amount;
}

size_t LoopInfo::get_increment() const {
    return m_increment;
}

std::vector<bool> LoopInfo::get_is_incremented() const {
    std::vector<bool> values;
    values.reserve(m_entry_points.size() + m_exit_points.size());
    iterate_through_ports([&values](const LoopPort& port) { values.push_back(port.is_incremented); });
    return values;
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
    iterate_through_ports([dim_idx](LoopPort& port) { port.dim_idx = dim_idx; });
}

template<>
std::vector<LoopPort>::iterator LoopInfo::find_loop_port(const LoopPort& loop_port) {
    auto& ports = loop_port.expr_port->get_type() == ExpressionPort::Input ? m_entry_points : m_exit_points;
    const auto it = std::find_if(ports.begin(), ports.end(),
                                 [&loop_port](const LoopPort& point) { return point == loop_port; });
    OPENVINO_ASSERT(it != ports.end(), "Failed update_loop_port: existing loop ports has not been found");
    return it;
}

template<>
std::vector<LoopPort>::iterator LoopInfo::find_loop_port(const ExpressionPort& expr_port) {
    auto& ports = expr_port.get_type() == ExpressionPort::Input ? m_entry_points : m_exit_points;
    const auto it = std::find_if(ports.begin(), ports.end(),
                                [&expr_port](const LoopPort& point) { return *point.expr_port.get() == expr_port; });
    return it;
}

void LoopInfo::replace_with_new_ports(const LoopPort& actual_port, const std::vector<LoopPort>& target_ports) {
    auto& ports = actual_port.expr_port->get_type() == ExpressionPort::Input ? m_entry_points : m_exit_points;
    auto port_it = find_loop_port(actual_port);
    port_it = ports.erase(port_it);
    ports.insert(port_it, target_ports.cbegin(), target_ports.cend());
}

void LoopInfo::replace_with_new_ports(const ExpressionPort& actual_port, const std::vector<ExpressionPort>& target_ports) {
    auto& ports = actual_port.get_type() == ExpressionPort::Input ? m_entry_points : m_exit_points;
    auto port_it = find_loop_port(actual_port);
    // In some cases actual ExpressionPort may not be LoopPort. We shouldn't throw exception here since ExpressionPort is not strong condition as LoopPort
    // For example, not all inner loop ports are ports of outer loops
    if (port_it == ports.end())
        return;

    // to save other parameters except expression port
    std::vector<LoopPort> target_loop_ports(target_ports.size(), *port_it);
    std::transform(target_loop_ports.begin(), target_loop_ports.end(), target_ports.begin(), target_loop_ports.begin(),
                   [](LoopPort loop_port, const ExpressionPort& expr_port) {
                       LoopPort copy = std::move(loop_port);  // to save loop port parameters
                       copy.expr_port = std::make_shared<ExpressionPort>(expr_port);
                       return copy;
                   });
    port_it = ports.erase(port_it);
    ports.insert(port_it, target_loop_ports.cbegin(), target_loop_ports.cend());
}

std::vector<LoopPort> LoopInfo::clone_loop_ports(const ExpressionMap& expr_map, const std::vector<LoopPort>& loop_ports) {
    std::vector<LoopPort> cloned_port_points;
    cloned_port_points.reserve(loop_ports.size());
    for (const auto& p : loop_ports) {
        const auto& expr = p.expr_port->get_expr().get();
        OPENVINO_ASSERT(expr_map.count(expr), "Can't clone LoopInfo: old expression is not in the map");
        const auto& new_expr = expr_map.at(expr);
        cloned_port_points.emplace_back(*p.clone_with_new_expr(new_expr));
    }
    return cloned_port_points;
}

UnifiedLoopInfo::UnifiedLoopInfo(size_t work_amount, size_t increment,
                                 const std::vector<LoopPort>& entries, const std::vector<LoopPort>& exits,
                                 const SpecificIterationHandlers& handlers)
    : LoopInfo(work_amount, increment, entries, exits), m_handlers(handlers),
      m_entry_port_descs(std::vector<LoopPortDesc>(entries.size())),
      m_exit_port_descs(std::vector<LoopPortDesc>(exits.size())) {}

UnifiedLoopInfo::UnifiedLoopInfo(size_t work_amount, size_t increment,
                                 const std::vector<ExpressionPort>& entries, const std::vector<ExpressionPort>& exits,
                                 const SpecificIterationHandlers& handlers)
    : LoopInfo(work_amount, increment, entries, exits), m_handlers(handlers),
      m_entry_port_descs(std::vector<LoopPortDesc>(entries.size())),
      m_exit_port_descs(std::vector<LoopPortDesc>(exits.size())) {}

UnifiedLoopInfo::UnifiedLoopInfo(size_t work_amount, size_t increment,
                                 const std::vector<LoopPort>& entries, const std::vector<LoopPort>& exits,
                                 const std::vector<LoopPortDesc>& in_shifts, const std::vector<LoopPortDesc>& out_shifts,
                                 const SpecificIterationHandlers& handlers)
    : LoopInfo(work_amount, increment, entries, exits), m_handlers(handlers), m_entry_port_descs(in_shifts), m_exit_port_descs(out_shifts) {}

std::shared_ptr<LoopInfo> UnifiedLoopInfo::clone_with_new_expr(const ExpressionMap& expr_map) const {
    const auto& new_entry_points = clone_loop_ports(expr_map, m_entry_points);
    const auto& new_exit_points = clone_loop_ports(expr_map, m_exit_points);

    return std::make_shared<UnifiedLoopInfo>(m_work_amount, m_increment, new_entry_points, new_exit_points,
                                             m_entry_port_descs, m_exit_port_descs, m_handlers);
}

const SpecificIterationHandlers& UnifiedLoopInfo::get_handlers() const {
    return m_handlers;
}

std::vector<int64_t> UnifiedLoopInfo::get_ptr_increments() const {
    std::vector<int64_t> values;
    values.reserve(m_entry_points.size() + m_exit_points.size());
    iterate_through_port_descs([&values](const LoopPortDesc& shift) { values.push_back(shift.ptr_increment); });
    return values;
}

std::vector<int64_t> UnifiedLoopInfo::get_finalization_offsets() const {
    std::vector<int64_t> values;
    values.reserve(m_entry_points.size() + m_exit_points.size());
    iterate_through_port_descs([&values](const LoopPortDesc& shift) { values.push_back(shift.finalization_offset); });
    return values;
}

std::vector<int64_t> UnifiedLoopInfo::get_data_sizes() const {
    std::vector<int64_t> values;
    values.reserve(m_entry_points.size() + m_exit_points.size());
    iterate_through_port_descs([&values](const LoopPortDesc& shift) { values.push_back(shift.data_size); });
    return values;
}

void UnifiedLoopInfo::set_handlers(SpecificIterationHandlers handlers) {
    m_handlers = std::move(handlers);
}

const std::vector<UnifiedLoopInfo::LoopPortDesc>& UnifiedLoopInfo::get_entry_port_descs() const {
    return m_entry_port_descs;
}

const std::vector<UnifiedLoopInfo::LoopPortDesc>& UnifiedLoopInfo::get_exit_port_descs() const {
    return m_exit_port_descs;
}

std::vector<UnifiedLoopInfo::LoopPortInfo> UnifiedLoopInfo::get_entry_ports_info() const {
    OPENVINO_ASSERT(m_entry_points.size() == m_entry_port_descs.size(), "Incompatible count of entry point and descs");
    std::vector<UnifiedLoopInfo::LoopPortInfo> info(get_entry_count());
    for (size_t i = 0; i < get_entry_count(); ++i)
        info[i] = std::make_pair(m_entry_points[i], m_entry_port_descs[i]);
    return info;
}

std::vector<UnifiedLoopInfo::LoopPortInfo> UnifiedLoopInfo::get_exit_ports_info() const {
    OPENVINO_ASSERT(m_exit_points.size() == m_exit_port_descs.size(), "Incompatible count of exit point and descs");
    std::vector<UnifiedLoopInfo::LoopPortInfo> info(get_exit_count());
    for (size_t i = 0; i < get_exit_count(); ++i)
        info[i] = std::make_pair(m_exit_points[i], m_exit_port_descs[i]);
    return info;
}

namespace {
template<typename T>
void order(const std::vector<size_t>& new_order, std::vector<T>& values) {
    OPENVINO_ASSERT(new_order.size() == values.size(),
                    "Failed to sort values: `new_order` must contain new indexes for ALL values");
    OPENVINO_ASSERT(std::set<size_t>(new_order.cbegin(), new_order.cend()).size() == new_order.size(),
                    "Failed to sort values: new order must contain unique indexes");
    std::vector<T> ordered_values(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        ordered_values[new_order[i]] = values[i];
    }
    values = std::move(ordered_values);
}
}  // namespace

void UnifiedLoopInfo::sort_entry_ports(const std::vector<size_t>& new_order) {
    order(new_order, m_entry_points);
    order(new_order, m_entry_port_descs);
}

void UnifiedLoopInfo::sort_exit_ports(const std::vector<size_t>& new_order) {
    order(new_order, m_exit_points);
    order(new_order, m_exit_port_descs);
}

void UnifiedLoopInfo::replace_with_new_ports(const LoopPort& actual_port, const std::vector<LoopPort>& target_ports) {
    const auto is_entry = actual_port.expr_port->get_type() == ExpressionPort::Input;
    auto& ports = is_entry ? m_entry_points : m_exit_points;
    auto port_it = find_loop_port(actual_port);

    replace_with_new_port_descs(actual_port, std::distance(ports.begin(), port_it), target_ports, is_entry);

    // Instead of call `LoopInfo::replace_with_new_ports()`, we can explicitly remove LoopPort since we already found port_it.
    port_it = ports.erase(port_it);
    ports.insert(port_it, target_ports.cbegin(), target_ports.cend());
}

void UnifiedLoopInfo::replace_with_new_ports(const ExpressionPort& actual_port, const std::vector<ExpressionPort>& target_ports) {
    const auto is_entry = actual_port.get_type() == ExpressionPort::Input;
    auto& ports = is_entry ? m_entry_points : m_exit_points;
    auto port_it = std::find_if(ports.begin(), ports.end(),
                                [&actual_port](const LoopPort& point) { return *point.expr_port.get() == actual_port; });
    // In some cases actual ExpressionPort may not be LoopPort. We shouldn't throw exception here since ExpressionPort is not strong condition as LoopPort
    // For example, not all inner loop ports are ports of outer loops
    if (port_it == ports.end())
        return;

    replace_with_new_port_descs(actual_port, std::distance(ports.begin(), port_it), target_ports, is_entry);
    LoopInfo::replace_with_new_ports(actual_port, target_ports);
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

const pass::PassPipeline& ExpandedLoopInfo::get_handler_passes() const {
    switch (m_type) {
        case SpecificLoopIterType::FIRST_ITER:
            return get_unified_loop_info()->get_handlers().get_passes<SpecificLoopIterType::FIRST_ITER>();
        case SpecificLoopIterType::MAIN_BODY:
            return get_unified_loop_info()->get_handlers().get_passes<SpecificLoopIterType::MAIN_BODY>();
        case SpecificLoopIterType::LAST_ITER:
            return get_unified_loop_info()->get_handlers().get_passes<SpecificLoopIterType::LAST_ITER>();
        default:
            OPENVINO_THROW("Unknown SpecificLoopIterType");
    }
}

const std::vector<int64_t>& ExpandedLoopInfo::get_ptr_increments() const {
    return m_ptr_increments;
}

const std::vector<int64_t>& ExpandedLoopInfo::get_finalization_offsets() const {
    return m_finalization_offsets;
}

const std::vector<int64_t>& ExpandedLoopInfo::get_data_sizes() const {
    return m_data_sizes;
}

void ExpandedLoopInfo::replace_with_new_ports(const LoopPort& actual_port, const std::vector<LoopPort>& target_ports) {
    OPENVINO_ASSERT(target_ports.size() == 1, "ExpandedLoopInfo supports replace one port with only one port!");
    LoopInfo::replace_with_new_ports(actual_port, target_ports);
}

void ExpandedLoopInfo::replace_with_new_ports(const ExpressionPort& actual_port, const std::vector<ExpressionPort>& target_ports) {
    OPENVINO_ASSERT(target_ports.size() == 1, "ExpandedLoopInfo supports replace one port with only one port!");
    LoopInfo::replace_with_new_ports(actual_port, target_ports);
}

} // namespace lowered
} // namespace snippets
} // namespace ov
