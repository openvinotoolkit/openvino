// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/loop_info.hpp"

#include "snippets/utils.hpp"


namespace ov {
namespace snippets {
namespace lowered {

LoopInfo::LoopInfo(size_t work_amount, size_t increment, const std::vector<LoopPort>& entries, const std::vector<LoopPort>& exits, bool is_wa_const)
    : m_work_amount(work_amount), m_increment(increment), m_input_ports(entries), m_output_ports(exits), m_is_work_amount_const(is_wa_const) {}

LoopInfo::LoopInfo(size_t work_amount, size_t increment, const std::vector<ExpressionPort>& entries, const std::vector<ExpressionPort>& exits,
                   bool is_wa_const)
    : m_work_amount(work_amount), m_increment(increment), m_is_work_amount_const(is_wa_const) {
    m_input_ports.reserve(entries.size());
    m_output_ports.reserve(exits.size());
    for (const auto& port : entries)
        m_input_ports.emplace_back(port);
    for (const auto& port : exits)
        m_output_ports.emplace_back(port);
}

bool LoopInfo::is_dynamic() const {
    return utils::is_dynamic_value(m_work_amount) || utils::is_dynamic_value(m_increment);
}

size_t LoopInfo::get_dim_idx() const {
    OPENVINO_ASSERT(!m_input_ports.empty(), "Loop info must have at least one input port");
    auto equal_dim_idxes = [&](const LoopPort& p) {
        return p.dim_idx == m_input_ports[0].dim_idx;
    };
    if (std::all_of(m_input_ports.begin(), m_input_ports.end(), equal_dim_idxes) &&
        std::all_of(m_output_ports.begin(), m_output_ports.end(), equal_dim_idxes)) {
        return m_input_ports[0].dim_idx;
    } else {
        return UNDEFINED_DIM_IDX;
    }
}

size_t LoopInfo::get_input_count() const {
    return m_input_ports.size();
}

size_t LoopInfo::get_output_count() const {
    return m_output_ports.size();
}

size_t LoopInfo::get_work_amount() const {
    return m_work_amount;
}

size_t LoopInfo::get_increment() const {
    return m_increment;
}

std::vector<bool> LoopInfo::get_is_incremented() const {
    std::vector<bool> values;
    values.reserve(get_input_count() + get_output_count());
    iterate_through_ports([&values](const LoopPort& port) { values.push_back(port.is_incremented); });
    return values;
}

const std::vector<LoopPort>& LoopInfo::get_input_ports() const {
    return m_input_ports;
}

const std::vector<LoopPort>& LoopInfo::get_output_ports() const {
    return m_output_ports;
}

bool LoopInfo::is_work_amount_const() const {
    return m_is_work_amount_const;
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

void LoopInfo::set_work_amount_const(bool value) {
    m_is_work_amount_const = value;
}

template<>
std::vector<LoopPort>::iterator LoopInfo::find_loop_port(const LoopPort& loop_port) {
    auto& ports = loop_port.expr_port->get_type() == ExpressionPort::Input ? m_input_ports : m_output_ports;
    const auto it = std::find_if(ports.begin(), ports.end(),
                                 [&loop_port](const LoopPort& port) { return port == loop_port; });
    OPENVINO_ASSERT(it != ports.end(), "Failed update_loop_port: existing loop port has not been found");
    return it;
}

template<>
std::vector<LoopPort>::iterator LoopInfo::find_loop_port(const ExpressionPort& expr_port) {
    auto& ports = expr_port.get_type() == ExpressionPort::Input ? m_input_ports : m_output_ports;
    const auto it = std::find_if(ports.begin(), ports.end(),
                                [&expr_port](const LoopPort& port) { return *port.expr_port.get() == expr_port; });
    return it;
}

void LoopInfo::replace_with_new_ports(const LoopPort& actual_port, const std::vector<LoopPort>& target_ports) {
    auto& ports = actual_port.expr_port->get_type() == ExpressionPort::Input ? m_input_ports : m_output_ports;
    auto port_it = find_loop_port(actual_port);
    port_it = ports.erase(port_it);
    ports.insert(port_it, target_ports.cbegin(), target_ports.cend());
}

void LoopInfo::replace_with_new_ports(const ExpressionPort& actual_port, const std::vector<ExpressionPort>& target_ports) {
    auto& ports = actual_port.get_type() == ExpressionPort::Input ? m_input_ports : m_output_ports;
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

bool UnifiedLoopInfo::LoopPortDesc::is_dynamic() const {
    return utils::is_dynamic_value(ptr_increment) || utils::is_dynamic_value(finalization_offset);
}

UnifiedLoopInfo::UnifiedLoopInfo(size_t work_amount, size_t increment,
                                 const std::vector<LoopPort>& entries, const std::vector<LoopPort>& exits,
                                 const SpecificIterationHandlers& handlers, bool is_wa_const)
    : LoopInfo(work_amount, increment, entries, exits, is_wa_const), m_handlers(handlers),
      m_input_port_descs(std::vector<LoopPortDesc>(entries.size())), m_output_port_descs(std::vector<LoopPortDesc>(exits.size())) {
    validate();
}

UnifiedLoopInfo::UnifiedLoopInfo(size_t work_amount, size_t increment,
                                 const std::vector<ExpressionPort>& entries, const std::vector<ExpressionPort>& exits,
                                 const SpecificIterationHandlers& handlers, bool is_wa_const)
    : LoopInfo(work_amount, increment, entries, exits, is_wa_const), m_handlers(handlers),
      m_input_port_descs(std::vector<LoopPortDesc>(entries.size())), m_output_port_descs(std::vector<LoopPortDesc>(exits.size())) {
    validate();
}

UnifiedLoopInfo::UnifiedLoopInfo(size_t work_amount, size_t increment,
                                 const std::vector<LoopPort>& entries, const std::vector<LoopPort>& exits,
                                 const std::vector<LoopPortDesc>& in_shifts, const std::vector<LoopPortDesc>& out_shifts,
                                 const SpecificIterationHandlers& handlers, bool is_wa_const)
    : LoopInfo(work_amount, increment, entries, exits, is_wa_const), m_handlers(handlers), m_input_port_descs(in_shifts), m_output_port_descs(out_shifts) {
    validate();
}

void UnifiedLoopInfo::validate() const {
    OPENVINO_ASSERT(m_input_ports.size() == m_input_port_descs.size(), "Incompatible count of input port and descs");
    OPENVINO_ASSERT(m_output_ports.size() == m_output_port_descs.size(), "Incompatible count of output port and descs");
}

std::shared_ptr<LoopInfo> UnifiedLoopInfo::clone_with_new_expr(const ExpressionMap& expr_map) const {
    const auto& new_input_ports = clone_loop_ports(expr_map, m_input_ports);
    const auto& new_output_ports = clone_loop_ports(expr_map, m_output_ports);

    return std::make_shared<UnifiedLoopInfo>(m_work_amount, m_increment, new_input_ports, new_output_ports,
                                             m_input_port_descs, m_output_port_descs, m_handlers, m_is_work_amount_const);
}

bool UnifiedLoopInfo::is_dynamic() const {
    return LoopInfo::is_dynamic() ||
           std::any_of(m_input_port_descs.cbegin(), m_input_port_descs.cend(), [](const LoopPortDesc& shift) { return shift.is_dynamic(); }) ||
           std::any_of(m_output_port_descs.cbegin(), m_output_port_descs.cend(), [](const LoopPortDesc& shift) { return shift.is_dynamic(); });
}

const SpecificIterationHandlers& UnifiedLoopInfo::get_handlers() const {
    return m_handlers;
}

std::vector<int64_t> UnifiedLoopInfo::get_ptr_increments() const {
    std::vector<int64_t> values;
    values.reserve(get_input_count() + get_output_count());
    iterate_through_descs([&values](const LoopPortDesc& shift) { values.push_back(shift.ptr_increment); });
    return values;
}

std::vector<int64_t> UnifiedLoopInfo::get_finalization_offsets() const {
    std::vector<int64_t> values;
    values.reserve(get_input_count() + get_output_count());
    iterate_through_descs([&values](const LoopPortDesc& shift) { values.push_back(shift.finalization_offset); });
    return values;
}

std::vector<int64_t> UnifiedLoopInfo::get_data_sizes() const {
    std::vector<int64_t> values;
    values.reserve(get_input_count() + get_output_count());
    iterate_through_descs([&values](const LoopPortDesc& shift) { values.push_back(shift.data_size); });
    return values;
}

void UnifiedLoopInfo::set_handlers(SpecificIterationHandlers handlers) {
    m_handlers = std::move(handlers);
}

const std::vector<UnifiedLoopInfo::LoopPortDesc>& UnifiedLoopInfo::get_input_port_descs() const {
    return m_input_port_descs;
}

const std::vector<UnifiedLoopInfo::LoopPortDesc>& UnifiedLoopInfo::get_output_port_descs() const {
    return m_output_port_descs;
}

std::vector<UnifiedLoopInfo::LoopPortInfo> UnifiedLoopInfo::get_input_ports_info() const {
    OPENVINO_ASSERT(m_input_ports.size() == m_input_port_descs.size(), "Incompatible count of input port and descs");
    std::vector<UnifiedLoopInfo::LoopPortInfo> info(get_input_count());
    for (size_t i = 0; i < get_input_count(); ++i)
        info[i] = { m_input_ports[i], m_input_port_descs[i] };
    return info;
}

std::vector<UnifiedLoopInfo::LoopPortInfo> UnifiedLoopInfo::get_output_ports_info() const {
    OPENVINO_ASSERT(m_output_ports.size() == m_output_port_descs.size(), "Incompatible count of output port and descs");
    std::vector<UnifiedLoopInfo::LoopPortInfo> info(get_output_count());
    for (size_t i = 0; i < get_output_count(); ++i)
        info[i] = { m_output_ports[i], m_output_port_descs[i] };
    return info;
}

namespace {
template<typename T>
void order(const std::vector<size_t>& new_order, std::vector<T>& values) {
    const auto order_set = std::set<size_t>(new_order.cbegin(), new_order.cend());
    OPENVINO_ASSERT(new_order.size() == values.size() && order_set.size() == values.size(),
                    "Failed to sort values: `new order` must contain unique indexes");
    OPENVINO_ASSERT(*order_set.begin() == 0 && *order_set.rbegin() == (values.size() - 1),
                    "Failed to sort values: `new_order` must contain new indexes for ALL values");
    std::vector<T> ordered_values(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        ordered_values[new_order[i]] = values[i];
    }
    values = std::move(ordered_values);
}
}  // namespace

void UnifiedLoopInfo::reorder_input_ports(const std::vector<size_t>& new_order) {
    validate();
    order(new_order, m_input_ports);
    order(new_order, m_input_port_descs);
}

void UnifiedLoopInfo::reorder_output_ports(const std::vector<size_t>& new_order) {
    validate();
    order(new_order, m_output_ports);
    order(new_order, m_output_port_descs);
}

void UnifiedLoopInfo::replace_with_cloned_descs(size_t actual_port_idx, size_t new_count, bool is_input) {
    auto& descs = is_input ? m_input_port_descs : m_output_port_descs;
    std::vector<LoopPortDesc> target_shifts(new_count, descs[actual_port_idx]);
    auto shift_it = descs.erase(descs.begin() + actual_port_idx);
    descs.insert(shift_it, target_shifts.cbegin(), target_shifts.cend());
}

void UnifiedLoopInfo::replace_with_new_ports(const LoopPort& actual_port, const std::vector<LoopPort>& target_ports) {
    const auto is_input = actual_port.expr_port->get_type() == ExpressionPort::Input;
    auto& ports = is_input ? m_input_ports : m_output_ports;
    auto port_it = find_loop_port(actual_port);

    replace_with_cloned_descs(std::distance(ports.begin(), port_it), target_ports.size(), is_input);

    // Instead of call `LoopInfo::replace_with_new_ports()`, we can explicitly remove LoopPort since we already found port_it.
    port_it = ports.erase(port_it);
    ports.insert(port_it, target_ports.cbegin(), target_ports.cend());
    // Verify that everything is valid after this change
    validate();
}

void UnifiedLoopInfo::replace_with_new_ports(const ExpressionPort& actual_port, const std::vector<ExpressionPort>& target_ports) {
    const auto is_input = actual_port.get_type() == ExpressionPort::Input;
    auto& ports = is_input ? m_input_ports : m_output_ports;
    auto port_it = std::find_if(ports.begin(), ports.end(),
                                [&actual_port](const LoopPort& port) { return *port.expr_port.get() == actual_port; });
    // In some cases actual ExpressionPort may not be LoopPort. We shouldn't throw exception here since ExpressionPort is not strong condition as LoopPort
    // For example, not all inner loop ports are ports of outer loops
    if (port_it == ports.end())
        return;

    replace_with_cloned_descs(std::distance(ports.begin(), port_it), target_ports.size(), is_input);
    LoopInfo::replace_with_new_ports(actual_port, target_ports);
    // Verify that everything is valid after this change
    validate();
}

ExpandedLoopInfo::ExpandedLoopInfo(size_t work_amount, size_t increment,
                                   const std::vector<LoopPort>& entries, const std::vector<LoopPort>& exits,
                                   std::vector<int64_t> ptr_increments, std::vector<int64_t> final_offsets, std::vector<int64_t> data_sizes,
                                   SpecificLoopIterType type, std::shared_ptr<UnifiedLoopInfo> unified_loop_info, bool is_wa_const)
    : LoopInfo(work_amount, increment, entries, exits, is_wa_const),
      m_ptr_increments(std::move(ptr_increments)), m_finalization_offsets(std::move(final_offsets)),
      m_data_sizes(std::move(data_sizes)), m_type(type), m_unified_loop_info(std::move(unified_loop_info)) {
    validate();
}

void ExpandedLoopInfo::validate() const {
    OPENVINO_ASSERT(m_unified_loop_info, "Failed to create ExpandedLoopInfo: unified loop info is nullptr!");
    const auto count = get_input_count() + get_output_count();
    OPENVINO_ASSERT(utils::everyone_is(count, m_ptr_increments.size(), m_finalization_offsets.size(), m_data_sizes.size()),
                    "Incompatible data ptr shifts!");
}

std::shared_ptr<LoopInfo> ExpandedLoopInfo::clone_with_new_expr(const ExpressionMap& expr_map) const {
    const auto& new_input_ports = clone_loop_ports(expr_map, m_input_ports);
    const auto& new_output_ports = clone_loop_ports(expr_map, m_output_ports);

    return std::make_shared<ExpandedLoopInfo>(m_work_amount, m_increment, new_input_ports, new_output_ports,
                                              m_ptr_increments, m_finalization_offsets, m_data_sizes, m_type, m_unified_loop_info, m_is_work_amount_const);
}

bool ExpandedLoopInfo::is_dynamic() const {
    return LoopInfo::is_dynamic() ||
           std::any_of(m_ptr_increments.cbegin(), m_ptr_increments.cend(), [](size_t v) { return utils::is_dynamic_value(v); }) ||
           std::any_of(m_finalization_offsets.cbegin(), m_finalization_offsets.cend(), [](size_t v) { return utils::is_dynamic_value(v); });
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

void ExpandedLoopInfo::update_ptr_increments(const std::vector<int64_t>& new_values) {
    OPENVINO_ASSERT(new_values.size() == m_ptr_increments.size(), "Failed to update ptr_increments: incompatible counts");
    m_ptr_increments.assign(new_values.cbegin(), new_values.end());
}

void ExpandedLoopInfo::update_finalization_offsets(const std::vector<int64_t>& new_values) {
    OPENVINO_ASSERT(new_values.size() == m_finalization_offsets.size(), "Failed to update finalization_offsets: incompatible counts");
    m_finalization_offsets.assign(new_values.cbegin(), new_values.end());
}

void ExpandedLoopInfo::replace_with_new_ports(const LoopPort& actual_port, const std::vector<LoopPort>& target_ports) {
    OPENVINO_ASSERT(target_ports.size() == 1, "ExpandedLoopInfo supports replace one port with only one port!");
    LoopInfo::replace_with_new_ports(actual_port, target_ports);
    validate();
}

void ExpandedLoopInfo::replace_with_new_ports(const ExpressionPort& actual_port, const std::vector<ExpressionPort>& target_ports) {
    OPENVINO_ASSERT(target_ports.size() == 1, "ExpandedLoopInfo supports replace one port with only one port!");
    LoopInfo::replace_with_new_ports(actual_port, target_ports);
    validate();
}

} // namespace lowered
} // namespace snippets
} // namespace ov
