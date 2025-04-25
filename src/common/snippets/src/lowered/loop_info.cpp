// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/loop_info.hpp"

#include "snippets/utils/utils.hpp"


namespace ov {
namespace snippets {
namespace lowered {

LoopInfo::LoopInfo(size_t work_amount, size_t increment, const std::vector<LoopPort>& entries, const std::vector<LoopPort>& exits)
    : m_work_amount(work_amount), m_increment(increment), m_input_ports(entries), m_output_ports(exits) {}

LoopInfo::LoopInfo(size_t work_amount, size_t increment, const std::vector<ExpressionPort>& entries, const std::vector<ExpressionPort>& exits)
    : m_work_amount(work_amount), m_increment(increment) {
    m_input_ports.reserve(entries.size());
    m_output_ports.reserve(exits.size());
    for (const auto& port : entries)
        m_input_ports.push_back(LoopPort::create<LoopPort::Type::Incremented>(port));
    for (const auto& port : exits)
        m_output_ports.push_back(LoopPort::create<LoopPort::Type::Incremented>(port));
}

bool LoopInfo::is_dynamic() const {
    return utils::is_dynamic_value(m_work_amount) || utils::is_dynamic_value(m_increment);
}

size_t LoopInfo::get_dim_idx() const {
    OPENVINO_ASSERT(!m_input_ports.empty(), "Loop info must have at least one input port");

    auto is_processed = [](const LoopPort& p) { return p.is_processed(); };
    auto is_processed_it = std::find_if(m_input_ports.begin(), m_input_ports.end(), is_processed);
    if (is_processed_it == m_input_ports.end()) {
        is_processed_it = std::find_if(m_output_ports.begin(), m_output_ports.end(), is_processed);
        if (is_processed_it == m_output_ports.end())
            return LoopPort::UNDEFINED_DIM_IDX;
    }
    const auto dim_idx = is_processed_it->get_dim_idx();

    auto equal_dim_idxes = [&](const LoopPort& p) { return !p.is_processed() || p.get_dim_idx() == dim_idx; };
    if (std::all_of(m_input_ports.begin(), m_input_ports.end(), equal_dim_idxes) &&
        std::all_of(m_output_ports.begin(), m_output_ports.end(), equal_dim_idxes)) {
        return dim_idx;
    } else {
        return LoopPort::UNDEFINED_DIM_IDX;
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
    iterate_through_ports([&values](const LoopPort& port) { values.push_back(port.is_incremented()); });
    return values;
}

const std::vector<LoopPort>& LoopInfo::get_input_ports() const {
    return m_input_ports;
}

const std::vector<LoopPort>& LoopInfo::get_output_ports() const {
    return m_output_ports;
}

void LoopInfo::set_work_amount(size_t work_amount) {
    m_work_amount = work_amount;
}

void LoopInfo::set_increment(size_t increment) {
    m_increment = increment;
}

void LoopInfo::set_dim_idx(size_t dim_idx) {
    auto setter = [dim_idx](LoopPort& port) { if (port.is_processed()) port.set_dim_idx(dim_idx); };
    std::for_each(m_input_ports.begin(), m_input_ports.end(), setter);
    std::for_each(m_output_ports.begin(), m_output_ports.end(), setter);
}

template<>
std::vector<LoopPort>::iterator LoopInfo::find_loop_port(const LoopPort& loop_port) {
    auto& ports = loop_port.get_expr_port()->get_type() == ExpressionPort::Input ? m_input_ports : m_output_ports;
    const auto it = std::find_if(ports.begin(), ports.end(),
                                 [&loop_port](const LoopPort& port) { return port == loop_port; });
    OPENVINO_ASSERT(it != ports.end(), "Failed find_loop_port: existing loop port has not been found");
    return it;
}

template<>
std::vector<LoopPort>::iterator LoopInfo::find_loop_port(const ExpressionPort& expr_port) {
    auto& ports = expr_port.get_type() == ExpressionPort::Input ? m_input_ports : m_output_ports;
    const auto it = std::find_if(ports.begin(), ports.end(),
                                [&expr_port](const LoopPort& port) { return *port.get_expr_port() == expr_port; });
    return it;
}

bool LoopInfo::is_loop_port(const ExpressionPort& expr_port) {
    const auto& loop_port_it = find_loop_port(expr_port);
    const auto& ports = expr_port.get_type() == ExpressionPort::Input ? m_input_ports : m_output_ports;
    return loop_port_it != ports.end();
}

const LoopPort& LoopInfo::get_loop_port(const ExpressionPort& expr_port) {
    OPENVINO_ASSERT(is_loop_port(expr_port), "Failed get_loop_port: expr_port is not a loop port");
    return *find_loop_port(expr_port);
}

namespace {
void validate_new_target_ports(const std::vector<LoopPort>& target_ports, ExpressionPort::Type target_type) {
    OPENVINO_ASSERT(target_ports.empty() ||
                    std::all_of(target_ports.cbegin(), target_ports.cend(),
                                [&target_type](const LoopPort& target_port) { return target_type == target_port.get_expr_port()->get_type(); }));
}
void validate_new_target_ports(const std::vector<ExpressionPort>& target_ports, ExpressionPort::Type target_type) {
    OPENVINO_ASSERT(target_ports.empty() ||
                    std::all_of(target_ports.cbegin(), target_ports.cend(),
                                [&target_type](const ExpressionPort& target_port) { return target_type == target_port.get_type(); }));
}
} // namespace

void LoopInfo::replace_with_new_ports(const LoopPort& actual_port, const std::vector<LoopPort>& target_ports) {
    const auto& actual_port_type = actual_port.get_expr_port()->get_type();
    validate_new_target_ports(target_ports, actual_port_type);

    auto& ports = actual_port_type == ExpressionPort::Input ? m_input_ports : m_output_ports;
    auto port_it = find_loop_port(actual_port);
    port_it = ports.erase(port_it);
    ports.insert(port_it, target_ports.cbegin(), target_ports.cend());
}

void LoopInfo::replace_with_new_ports(const ExpressionPort& actual_port, const std::vector<ExpressionPort>& target_ports) {
    const auto& actual_port_type = actual_port.get_type();
    validate_new_target_ports(target_ports, actual_port_type);

    auto& ports = actual_port_type == ExpressionPort::Input ? m_input_ports : m_output_ports;
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
                       copy.set_expr_port(std::make_shared<ExpressionPort>(expr_port));
                       return copy;
                   });
    port_it = ports.erase(port_it);
    ports.insert(port_it, target_loop_ports.cbegin(), target_loop_ports.cend());
}

std::vector<LoopPort> LoopInfo::clone_loop_ports(const ExpressionMap& expr_map, const std::vector<LoopPort>& loop_ports) {
    std::vector<LoopPort> cloned_port_points;
    cloned_port_points.reserve(loop_ports.size());
    for (const auto& p : loop_ports) {
        const auto& expr = p.get_expr_port()->get_expr().get();
        OPENVINO_ASSERT(expr_map.count(expr), "Can't clone LoopInfo: old expression is not in the map");
        const auto& new_expr = expr_map.at(expr);
        cloned_port_points.emplace_back(*p.clone_with_new_expr(new_expr));
    }
    return cloned_port_points;
}

bool UnifiedLoopInfo::LoopPortDesc::is_dynamic() const {
    return utils::is_dynamic_value(ptr_increment) || utils::is_dynamic_value(finalization_offset);
}

bool UnifiedLoopInfo::LoopPortDesc::is_static() const {
    return !is_dynamic();
}

bool operator==(const UnifiedLoopInfo::LoopPortDesc& lhs, const UnifiedLoopInfo::LoopPortDesc& rhs) {
    if (&lhs == &rhs)
        return true;
    return lhs.ptr_increment == rhs.ptr_increment && lhs.finalization_offset == rhs.finalization_offset && lhs.data_size == rhs.data_size;
}
bool operator!=(const UnifiedLoopInfo::LoopPortDesc& lhs, const UnifiedLoopInfo::LoopPortDesc& rhs) {
    return !(rhs == lhs);
}

UnifiedLoopInfo::UnifiedLoopInfo(size_t work_amount, size_t increment,
                                 const std::vector<LoopPort>& entries, const std::vector<LoopPort>& exits,
                                 const SpecificIterationHandlers& handlers)
    : LoopInfo(work_amount, increment, entries, exits), m_handlers(handlers),
      m_input_port_descs(std::vector<LoopPortDesc>(entries.size())), m_output_port_descs(std::vector<LoopPortDesc>(exits.size())) {
    sort_ports();
}

UnifiedLoopInfo::UnifiedLoopInfo(size_t work_amount, size_t increment,
                                 const std::vector<ExpressionPort>& entries, const std::vector<ExpressionPort>& exits,
                                 const SpecificIterationHandlers& handlers)
    : LoopInfo(work_amount, increment, entries, exits), m_handlers(handlers),
      m_input_port_descs(std::vector<LoopPortDesc>(entries.size())), m_output_port_descs(std::vector<LoopPortDesc>(exits.size())) {
    sort_ports();
}

UnifiedLoopInfo::UnifiedLoopInfo(size_t work_amount, size_t increment,
                                 const std::vector<LoopPort>& entries, const std::vector<LoopPort>& exits,
                                 const std::vector<LoopPortDesc>& in_shifts, const std::vector<LoopPortDesc>& out_shifts,
                                 const SpecificIterationHandlers& handlers)
    : LoopInfo(work_amount, increment, entries, exits), m_handlers(handlers), m_input_port_descs(in_shifts), m_output_port_descs(out_shifts) {
    sort_ports();
}

std::shared_ptr<LoopInfo> UnifiedLoopInfo::clone_with_new_expr(const ExpressionMap& expr_map, LoopInfoMap& loop_map) const {
    if (loop_map.count(this) == 0) {
        const auto& new_input_ports = clone_loop_ports(expr_map, m_input_ports);
        const auto& new_output_ports = clone_loop_ports(expr_map, m_output_ports);

        loop_map[this] = std::make_shared<UnifiedLoopInfo>(m_work_amount, m_increment, new_input_ports, new_output_ports,
                                                           m_input_port_descs, m_output_port_descs, m_handlers);
    }
    return loop_map.at(this);
}

void UnifiedLoopInfo::apply(const std::function<void(const LoopInfoPtr&)>& func, LoopInfoSet& applied_loops) {
    if (applied_loops.count(this) == 0) {
        func(shared_from_this());
        applied_loops.insert(this);
    }
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
        ordered_values[i] = values[new_order[i]];
    }
    values = std::move(ordered_values);
}
std::vector<size_t> get_port_index_order(const std::vector<LoopPort>& ports) {
    std::vector<size_t> new_indexes(ports.size());
    std::iota(new_indexes.begin(), new_indexes.end(), 0);
    std::sort(new_indexes.begin(), new_indexes.end(),
        [ports](size_t l, size_t r) {
            const auto& expr_port_l = ports[l].get_expr_port();
            const auto& expr_port_r = ports[r].get_expr_port();
            if (expr_port_l->get_expr() == expr_port_r->get_expr())
                return expr_port_l->get_index() < expr_port_r->get_index();
            return expr_port_l->get_expr()->get_exec_num() < expr_port_r->get_expr()->get_exec_num();
        });
    return new_indexes;
}
}  // namespace

void UnifiedLoopInfo::sort_ports() {
    OPENVINO_ASSERT(m_input_ports.size() == m_input_port_descs.size(), "Incompatible count of input port and descs");
    OPENVINO_ASSERT(m_output_ports.size() == m_output_port_descs.size(), "Incompatible count of output port and descs");

    auto reorder = [](std::vector<LoopPort>& ports, std::vector<LoopPortDesc>& descs) {
        if (!ports.empty()) {
            const auto new_order = get_port_index_order(ports);
            order(new_order, ports);
            order(new_order, descs);
        }
    };
    reorder(m_input_ports, m_input_port_descs);
    reorder(m_output_ports, m_output_port_descs);
}

UnifiedLoopInfo::LoopPortInfo UnifiedLoopInfo::get_loop_port_info(const ExpressionPort& expr_port) {
    OPENVINO_ASSERT(is_loop_port(expr_port), "Failed get_loop_port: expr_port is not a loop port");
    const auto is_input = expr_port.get_type() == ExpressionPort::Input;
    const auto& ports = is_input ? m_input_ports : m_output_ports;
    const auto& descs = is_input ? m_input_port_descs : m_output_port_descs;
    const auto it = std::find_if(ports.begin(), ports.end(),
                                [&expr_port](const LoopPort& port) { return *port.get_expr_port() == expr_port; });
    const auto index = static_cast<size_t>(std::distance(ports.cbegin(), it));
    OPENVINO_ASSERT(index < ports.size() && index < descs.size(), "LoopPortInfo has not been found!");
    return {ports[index], descs[index]};
}

void UnifiedLoopInfo::replace_with_cloned_descs(size_t actual_port_idx, size_t new_count, bool is_input) {
    auto& descs = is_input ? m_input_port_descs : m_output_port_descs;
    std::vector<LoopPortDesc> target_shifts(new_count, descs[actual_port_idx]);
    auto shift_it = descs.erase(descs.begin() + actual_port_idx);
    descs.insert(shift_it, target_shifts.cbegin(), target_shifts.cend());
}

void UnifiedLoopInfo::replace_with_new_ports(const LoopPort& actual_port, const std::vector<LoopPort>& target_ports) {
    const auto& actual_port_type = actual_port.get_expr_port()->get_type();
    validate_new_target_ports(target_ports, actual_port_type);

    const auto is_input = actual_port.get_expr_port()->get_type() == ExpressionPort::Input;
    auto& ports = is_input ? m_input_ports : m_output_ports;
    auto port_it = find_loop_port(actual_port);

    replace_with_cloned_descs(std::distance(ports.begin(), port_it), target_ports.size(), is_input);

    // Instead of call `LoopInfo::replace_with_new_ports()`, we can explicitly remove LoopPort since we already found port_it.
    port_it = ports.erase(port_it);
    ports.insert(port_it, target_ports.cbegin(), target_ports.cend());

    // Sort ports
    sort_ports();
}

void UnifiedLoopInfo::replace_with_new_ports(const ExpressionPort& actual_port, const std::vector<ExpressionPort>& target_ports) {
    const auto& actual_port_type = actual_port.get_type();
    validate_new_target_ports(target_ports, actual_port_type);

    const auto is_input = actual_port.get_type() == ExpressionPort::Input;
    auto& ports = is_input ? m_input_ports : m_output_ports;
    auto port_it = find_loop_port(actual_port);
    // In some cases actual ExpressionPort may not be LoopPort. We shouldn't throw exception here since ExpressionPort is not strong condition as LoopPort
    // For example, not all inner loop ports are ports of outer loops
    if (port_it == ports.end())
        return;

    replace_with_cloned_descs(std::distance(ports.begin(), port_it), target_ports.size(), is_input);
    LoopInfo::replace_with_new_ports(actual_port, target_ports);

    // Sort ports
    sort_ports();
}

InnerSplittedUnifiedLoopInfo::InnerSplittedUnifiedLoopInfo(size_t increment, const std::vector<LoopPort>& entries, const std::vector<LoopPort>& exits,
                                                           const std::vector<LoopPortDesc>& in_descs, const std::vector<LoopPortDesc>& out_descs,
                                                           const SpecificIterationHandlers& handlers, LoopInfoPtr outer_splitted_loop_info)
    : UnifiedLoopInfo(utils::get_dynamic_value<size_t>(), increment, entries, exits, in_descs, out_descs, handlers),
      m_outer_splitted_loop_info(std::move(outer_splitted_loop_info)) {
    OPENVINO_ASSERT(m_outer_splitted_loop_info != nullptr, "Outer Splitted Loop Info is missed!");
}

std::shared_ptr<LoopInfo> InnerSplittedUnifiedLoopInfo::clone_with_new_expr(const ExpressionMap& expr_map, LoopInfoMap& loop_map) const {
    if (loop_map.count(this) == 0) {
        auto cloned_outer_splitted_loop_info = m_outer_splitted_loop_info->clone_with_new_expr(expr_map, loop_map);
        const auto& new_input_ports = clone_loop_ports(expr_map, m_input_ports);
        const auto& new_output_ports = clone_loop_ports(expr_map, m_output_ports);

        loop_map[this] = std::make_shared<InnerSplittedUnifiedLoopInfo>(m_increment, new_input_ports, new_output_ports,
                                                                        m_input_port_descs, m_output_port_descs, m_handlers,
                                                                        std::move(cloned_outer_splitted_loop_info));
    }
    return loop_map.at(this);
}

void InnerSplittedUnifiedLoopInfo::apply(const std::function<void(const LoopInfoPtr&)>& func, LoopInfoSet& applied_loops) {
    if (applied_loops.count(this) == 0) {
        m_outer_splitted_loop_info->apply(func, applied_loops);
        func(shared_from_this());
        applied_loops.insert(this);
    }
}

size_t InnerSplittedUnifiedLoopInfo::get_work_amount() const {
    return get_outer_splitted_loop_info()->get_increment();
}

LoopInfoPtr InnerSplittedUnifiedLoopInfo::get_outer_splitted_loop_info() const {
    OPENVINO_ASSERT(m_outer_splitted_loop_info, "Outer Splitted loop info is nullptr!");
    return m_outer_splitted_loop_info;
}

void InnerSplittedUnifiedLoopInfo::set_work_amount(size_t work_amount) {
    OPENVINO_THROW("InnerSplittedUnifiedLoopInfo doesn't support `set_work_amount`");
}

void InnerSplittedUnifiedLoopInfo::set_outer_splitted_loop_info(LoopInfoPtr outer) {
    OPENVINO_ASSERT(outer, "Outer Splitted loop info cannot be nullptr!");
    m_outer_splitted_loop_info = std::move(outer);
}

ExpandedLoopInfo::ExpandedLoopInfo(size_t work_amount, size_t increment,
                                   const std::vector<LoopPort>& entries, const std::vector<LoopPort>& exits,
                                   std::vector<int64_t> ptr_increments, std::vector<int64_t> final_offsets, std::vector<int64_t> data_sizes,
                                   SpecificLoopIterType type, std::shared_ptr<UnifiedLoopInfo> unified_loop_info, bool evaluate_once)
    : LoopInfo(work_amount, increment, entries, exits),
      m_ptr_increments(std::move(ptr_increments)), m_finalization_offsets(std::move(final_offsets)),
      m_data_sizes(std::move(data_sizes)), m_type(type), m_unified_loop_info(std::move(unified_loop_info)), m_evaluate_once(evaluate_once) {
    OPENVINO_ASSERT(m_unified_loop_info, "Failed to create ExpandedLoopInfo: unified loop info is nullptr!");
    sort_ports();
}

std::shared_ptr<LoopInfo> ExpandedLoopInfo::clone_with_new_expr(const ExpressionMap& expr_map, LoopInfoMap& loop_map) const {
    if (loop_map.count(this) == 0) {
        auto cloned_unified_loop_info = ov::as_type_ptr<UnifiedLoopInfo>(m_unified_loop_info->clone_with_new_expr(expr_map, loop_map));
        const auto& new_input_ports = clone_loop_ports(expr_map, m_input_ports);
        const auto& new_output_ports = clone_loop_ports(expr_map, m_output_ports);

        loop_map[this] = std::make_shared<ExpandedLoopInfo>(m_work_amount, m_increment, new_input_ports, new_output_ports,
                                                            m_ptr_increments, m_finalization_offsets, m_data_sizes, m_type,
                                                            std::move(cloned_unified_loop_info), m_evaluate_once);
    }
    return loop_map.at(this);
}

void ExpandedLoopInfo::apply(const std::function<void(const LoopInfoPtr&)>& func, LoopInfoSet& applied_loops) {
    if (applied_loops.count(this) == 0) {
        m_unified_loop_info->apply(func, applied_loops);
        func(shared_from_this());
        applied_loops.insert(this);
    }
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

bool ExpandedLoopInfo::is_evaluate_once() const {
    return m_evaluate_once;
}

void ExpandedLoopInfo::set_evaluate_once(bool value) {
    m_evaluate_once = value;
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
    sort_ports();
}

void ExpandedLoopInfo::replace_with_new_ports(const ExpressionPort& actual_port, const std::vector<ExpressionPort>& target_ports) {
    OPENVINO_ASSERT(target_ports.size() == 1, "ExpandedLoopInfo supports replace one port with only one port!");
    LoopInfo::replace_with_new_ports(actual_port, target_ports);
    sort_ports();
}

namespace {
void order_subvector(const std::vector<size_t>& indexes, std::vector<int64_t>& original_vector, size_t count, size_t offset) {
    auto subvector = std::vector<int64_t>{original_vector.cbegin() + offset, original_vector.cbegin() + offset + count};
    order(indexes, subvector);
    std::copy(subvector.begin(), subvector.end(), original_vector.begin() + offset);
}
} // namespace

void ExpandedLoopInfo::sort_ports() {
    const auto count = get_input_count() + get_output_count();
    OPENVINO_ASSERT(utils::everyone_is(count, m_ptr_increments.size(), m_finalization_offsets.size(), m_data_sizes.size()),
                    "Incompatible data ptr shifts!");

    auto reorder = [this](std::vector<LoopPort>& ports, size_t count, size_t offset) {
        if (!ports.empty()) {
            const auto new_order = get_port_index_order(ports);
            order(new_order, ports);
            order_subvector(new_order, m_ptr_increments, count, offset);
            order_subvector(new_order, m_finalization_offsets, count, offset);
            order_subvector(new_order, m_data_sizes, count, offset);
        }
    };
    reorder(m_input_ports, get_input_count(), 0);
    reorder(m_output_ports, get_output_count(), get_input_count());
}

} // namespace lowered
} // namespace snippets
} // namespace ov
