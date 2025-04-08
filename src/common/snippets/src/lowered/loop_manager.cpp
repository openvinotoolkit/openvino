// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/loop_manager.hpp"

#include "snippets/lowered/expression.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/utils/utils.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/core/type.hpp"

#include "snippets/utils/loop_utils.hpp"
#include "snippets/itt.hpp"


namespace ov {
namespace snippets {
namespace lowered {

std::shared_ptr<LoopManager> LoopManager::clone_with_new_expr(const ExpressionMap& expr_map) const {
    auto new_loop_manager = std::make_shared<LoopManager>();
    // To fully cloned all LoopInfo we have to create this map [old LoopInfo -> cloned LoopInfo],
    // because some LoopInfo types contains pointer to another LoopInfo
    // so we should recurrently make a cloning of LoopInfos'
    LoopInfoMap loop_info_map; // [ old - > cloned ]
    for (const auto& id_info : m_map)
        new_loop_manager->m_map.insert({id_info.first, id_info.second->clone_with_new_expr(expr_map, loop_info_map)});
    new_loop_manager->next_id = next_id;
    return new_loop_manager;
}

size_t LoopManager::add_loop_info(const LoopInfoPtr& loop) {
    const auto index = next_id;
    m_map[index] = loop;
    next_id++;
    return index;
}

void LoopManager::remove_loop_info(size_t index) {
    m_map.erase(index);
}

const std::map<size_t, LoopInfoPtr>& LoopManager::get_map() const {
    return m_map;
}

std::vector<size_t> LoopManager::get_outer_expr_loops(const ExpressionPtr& expr, size_t loop_id) {
    const auto& loop_ids = expr->get_loop_ids();
    const auto it = std::find(loop_ids.cbegin(), loop_ids.cend(), loop_id);
    OPENVINO_ASSERT(it != loop_ids.cend(), "Loop ID hasn't been found");
    return std::vector<size_t>(loop_ids.cbegin(), it);
}

std::vector<size_t> LoopManager::get_common_outer_loops(const ExpressionPtr& lhs, const ExpressionPtr& rhs) {
    const auto& rhs_ids = rhs->get_loop_ids();
    const auto& lhs_ids = lhs->get_loop_ids();
    size_t idx = 0;
    while (idx < std::min(rhs_ids.size(), lhs_ids.size()) && rhs_ids[idx] == lhs_ids[idx]) {
        idx++;
    }
    return std::vector<size_t>(rhs_ids.cbegin(), rhs_ids.cbegin() + idx);
}

std::vector<size_t> LoopManager::get_common_outer_loops(const std::vector<ExpressionPtr>& exprs) {
    OPENVINO_ASSERT(!exprs.empty(), "Failed to find common outer loops for set of expressions: there no expressions");

    auto get_first_diff_id_idx = [](const std::vector<size_t>& lhs, const std::vector<size_t>& rhs) {
        size_t idx = 0;
        while (idx < std::min(lhs.size(), rhs.size()) && lhs[idx] == rhs[idx]) {
            idx++;
        }
        return idx;
    };

    const auto& first_loop_ids = exprs.front()->get_loop_ids();
    size_t common_idx = 0;
    for (size_t i = 1; i < exprs.size(); ++i) {
        common_idx = std::min(common_idx, get_first_diff_id_idx(first_loop_ids, exprs[i]->get_loop_ids()));
    }
    return std::vector<size_t>(first_loop_ids.cbegin(), first_loop_ids.cbegin() + common_idx);
}

std::pair<LinearIR::constExprIt, LinearIR::constExprIt> LoopManager::get_loop_bounds(const LinearIR &linear_ir, size_t loop_id) const {
    const auto loop_info = get_loop_info(loop_id);
    return get_loop_bounds(linear_ir, loop_id, loop_info->get_input_ports(), loop_info->get_output_ports());
}

std::pair<LinearIR::constExprIt, LinearIR::constExprIt> LoopManager::get_loop_bounds(const LinearIR &linear_ir, size_t loop_id,
                                                                                     const std::vector<LoopPort>& entries,
                                                                                     const std::vector<LoopPort>& exits) {
    OPENVINO_ASSERT(!entries.empty(), "Loop must have input ports");
    OPENVINO_ASSERT(!exits.empty(), "Loop must have output ports");

    const auto& entry_expr = entries.front().get_expr_port()->get_expr();
    auto loop_begin_pos = linear_ir.find(entry_expr);
    // Some operations in Loop can be before first input ports: Scalars, VectorBuffer.
    // We should iterate by them till the expr is in the corresponding Loop
    auto prev_loop_ids = (*std::prev(loop_begin_pos))->get_loop_ids();
    while (std::find(prev_loop_ids.begin(), prev_loop_ids.end(), loop_id) != prev_loop_ids.end()) {
        loop_begin_pos = std::prev(loop_begin_pos);
        prev_loop_ids = (*std::prev(loop_begin_pos))->get_loop_ids();
    }

    const auto& exit_expr = exits.back().get_expr_port()->get_expr();
    auto loop_end_pos = std::next(linear_ir.find_after(loop_begin_pos, exit_expr));
    // There might be LoopEnd with another `loop_id` but in the target Loop as well.
    auto current_loop_ids = (*loop_end_pos)->get_loop_ids();
    while (std::find(current_loop_ids.begin(), current_loop_ids.end(), loop_id) != current_loop_ids.end()) {
        loop_end_pos = std::next(loop_end_pos);
        current_loop_ids = (*loop_end_pos)->get_loop_ids();
    }

    // Check for the existing LoopBegin/LoopEnd
    if (const auto loop_end = ov::as_type_ptr<op::LoopEnd>((*loop_end_pos)->get_node())) {
        if (loop_end->get_id() == loop_id) {
            // loop_begin_pos is iterator of LoopBegin now
            // loop_end_pos is iterator of LoopEnd now
            loop_begin_pos = std::prev(loop_begin_pos);
            const auto loop_begin = loop_end->get_loop_begin();
            OPENVINO_ASSERT((*loop_begin_pos)->get_node() == loop_begin, "LoopBegin has not been found!");
        }
    }

    OPENVINO_ASSERT(loop_begin_pos != linear_ir.cend() && loop_end_pos != linear_ir.cend(), "Loop bounds haven't been found!");
    return std::make_pair(loop_begin_pos, loop_end_pos);
}

LoopPort LoopManager::get_loop_port_by_expr_port(const ExpressionPort& expr_port, const size_t loop_id) {
    const auto& loop_info = get_loop_info(loop_id);
    return loop_info->get_loop_port(expr_port);
}

void LoopManager::get_io_loop_ports(LinearIR::constExprIt loop_begin_pos,
                                    LinearIR::constExprIt loop_end_pos,
                                    std::vector<ExpressionPort> &entries,
                                    std::vector<ExpressionPort> &exits) {
    entries.clear();
    exits.clear();
    for (auto expr_it = loop_begin_pos; expr_it != loop_end_pos; ++expr_it) {
        const auto& expr = *expr_it;
        for (size_t i = 0; i < expr->get_input_count(); ++i) {
            const auto in_port = expr->get_input_port(i);
            const auto parent_expr = in_port.get_connected_ports().begin()->get_expr();
            if (!ov::is_type<ov::op::v0::Constant>(parent_expr->get_node()) &&
                std::find(loop_begin_pos, expr_it, parent_expr) == expr_it) {
                entries.push_back(in_port);
            }
        }
        for (size_t i = 0; i < expr->get_output_count(); ++i) {
            const auto out_port = expr->get_output_port(i);
            const auto consumer_ports = out_port.get_connected_ports();
            for (const auto& consumer : consumer_ports) {
                const auto& consumer_expr = consumer.get_expr();
                if (std::find(expr_it, loop_end_pos, consumer_expr) == loop_end_pos) {
                    exits.push_back(out_port);
                    break;
                }
            }
        }
    }
}

void LoopManager::mark_loop(LinearIR::constExprIt loop_begin_pos,
                            LinearIR::constExprIt loop_end_pos,
                            size_t loop_depth, size_t vector_size) {
    std::vector<ExpressionPort> loop_input_ports, loop_output_ports;
    LoopManager::get_io_loop_ports(loop_begin_pos, loop_end_pos, loop_input_ports, loop_output_ports);

    auto broadcast = [](std::vector<size_t>& lhs, const std::vector<size_t>& rhs, size_t index) -> void {
        if (rhs == lhs)
            return;
        const auto lhs_size = lhs.size();
        const auto rhs_size = rhs.size();
        const auto size = std::max(lhs_size, rhs_size);
        lhs.resize(size, 1);
        OPENVINO_ASSERT(index < size, "Incorrect index for broadcasting");
        const auto lhs_value = index < lhs_size ? *(lhs.crbegin() + index) : 1;
        const auto rhs_value = index < rhs_size ? *(rhs.crbegin() + index) : 1;
        OPENVINO_ASSERT(utils::broadcast_merge_dim(*(lhs.rbegin() + index), lhs_value, rhs_value),
                        "Failed to broadcast work amount in marking loop");
    };

    auto is_outside_loop = [](const std::vector<size_t>& subtensor) {
        return std::all_of(subtensor.begin(), subtensor.end(), utils::is_full_dim_value);
    };

    std::vector<size_t> loop_subtensor;
    std::vector<size_t> loop_tensor(loop_depth, 1);
    for (const auto& output_port : loop_output_ports) {
        const auto shape = utils::get_preordered_vdims(output_port);
        auto subtensor = output_port.get_descriptor_ptr()->get_subtensor();
        if (subtensor.empty()) {
            subtensor.resize(loop_depth, 1);
            subtensor[subtensor.size() - 1] = vector_size;
        }

        const size_t resizing_value = is_outside_loop(subtensor) ? utils::get_full_dim_value() : 1;
        while (subtensor.size() < loop_depth)
            subtensor.insert(subtensor.begin(), resizing_value);
        if (loop_subtensor.empty())
            loop_subtensor = subtensor;

        OPENVINO_ASSERT(std::equal(loop_subtensor.crbegin(), loop_subtensor.crbegin() + loop_depth, subtensor.crbegin()),
                        "Incorrect scheduling parameters for loop");

        for (size_t dim_idx = 0; dim_idx < loop_depth; ++dim_idx) {
            if (!utils::is_full_dim_value(*(subtensor.rbegin() + dim_idx))) {
                broadcast(loop_tensor, shape, dim_idx);
            }
        }
    }

    for (size_t dim_idx = 0; dim_idx < loop_depth; ++dim_idx) {
        OPENVINO_ASSERT(dim_idx < loop_subtensor.size(), "Incorrect indexes of Loop for markup");
        const auto& subtensor_value = *(loop_subtensor.rbegin() + dim_idx);
        if (utils::is_full_dim_value(subtensor_value)) {
            continue;
        }

        OPENVINO_ASSERT(dim_idx < loop_tensor.size(), "Incorrect indexes of Loop for markup");
        const auto work_amount = *(loop_tensor.rbegin() + dim_idx);
        const auto increment = subtensor_value;
        mark_loop(loop_begin_pos, loop_end_pos, work_amount, increment, dim_idx, loop_input_ports, loop_output_ports);
    }
}

size_t LoopManager::replace_with_new_loop(const LinearIR& linear_ir, LinearIR::constExprIt loop_begin_pos, LinearIR::constExprIt loop_end_pos,
                                          const LoopInfoPtr& loop_info, const size_t old_id) {
    const auto is_bound_explicit_loop_begin = ov::is_type<op::LoopBegin>(loop_begin_pos->get()->get_node());
    const auto is_bound_explicit_loop_end = ov::is_type<op::LoopEnd>(std::prev(loop_end_pos)->get()->get_node());
    OPENVINO_ASSERT((is_bound_explicit_loop_begin && is_bound_explicit_loop_end) || (!is_bound_explicit_loop_begin && !is_bound_explicit_loop_end),
                    "Incorrect LoopBounds!");
    const auto explicit_loop_bounds = is_bound_explicit_loop_begin && is_bound_explicit_loop_end;
    OPENVINO_ASSERT(std::all_of(m_map.cbegin(), m_map.cend(),
                    [&loop_info](const std::pair<size_t, LoopInfoPtr>& p) { return loop_info != p.second; }),
                    "Failed to replace with new Loop: this Loop already exists!");

    const auto old_loop_bounds = get_loop_bounds(linear_ir, old_id);

    const auto loop_id = this->add_loop_info(loop_info);
    const auto begin = explicit_loop_bounds ? std::next(loop_begin_pos) : loop_begin_pos;
    const auto end = explicit_loop_bounds ? std::prev(loop_end_pos) : loop_end_pos;
    for (auto expr_it = begin; expr_it != end; ++expr_it) {
        replace_loop_id(*expr_it, old_id, loop_id);
    }

    // If new bounds are equal to old loop bounds, this means that old Loop is removed totally from LIR
    // In this case old loop info must be completely removed from loop manager
    if (loop_begin_pos == old_loop_bounds.first && end == old_loop_bounds.second) {
        this->remove_loop_info(old_id);
    }
    return loop_id;
}

void LoopManager::fuse_loops(const LinearIR& linear_ir, size_t loop_id_upper, size_t loop_id_lower, bool fuse_into_upper) {
    const auto loop_bounds = get_loop_bounds(linear_ir, fuse_into_upper ? loop_id_lower : loop_id_upper);
    fuse_loops(loop_bounds.first, loop_bounds.second, loop_id_upper, loop_id_lower, fuse_into_upper);
}

void LoopManager::fuse_loops(LinearIR::constExprIt loop_begin_target, LinearIR::constExprIt loop_end_target,
                             size_t loop_id_upper, size_t loop_id_lower, bool fuse_into_upper) {
    const auto& loop_info_upper = get_loop_info<UnifiedLoopInfo>(loop_id_upper);
    const auto& loop_info_lower = get_loop_info<UnifiedLoopInfo>(loop_id_lower);

    auto input_ports_upper = loop_info_upper->get_input_ports();
    auto output_ports_upper = loop_info_upper->get_output_ports();
    auto input_ports_lower = loop_info_lower->get_input_ports();
    const auto& output_ports_lower = loop_info_lower->get_output_ports();
    fuse_loop_ports(output_ports_upper, input_ports_lower, loop_id_upper);

    const auto& from = fuse_into_upper ? loop_id_lower : loop_id_upper;
    const auto& to = fuse_into_upper ? loop_id_upper : loop_id_lower;

    // Since fusion can be called for broadcastable loops (one of the loops has work_amount = increment = 1),
    // maximum value is set to the fused loop
    const auto work_amount = std::max(loop_info_upper->get_work_amount(), loop_info_lower->get_work_amount());
    const auto increment = std::max(loop_info_upper->get_increment(), loop_info_lower->get_increment());
    const auto handlers = SpecificIterationHandlers::merge_handlers(loop_info_upper->get_handlers(), loop_info_lower->get_handlers());

    auto new_entries = std::move(input_ports_upper);
    new_entries.insert(new_entries.end(), input_ports_lower.begin(), input_ports_lower.end());
    auto new_exits = std::move(output_ports_upper);
    new_exits.insert(new_exits.end(), output_ports_lower.begin(), output_ports_lower.end());

    m_map[to] = std::make_shared<UnifiedLoopInfo>(work_amount, increment, new_entries, new_exits, handlers);

    // Need to handle InnerSplittedLoopInfo - update outer splitted loop info if it was fused
    for (const auto& p : m_map) {
        if (const auto inner_splitted_loop_info = ov::as_type_ptr<InnerSplittedUnifiedLoopInfo>(p.second)) {
            const auto outer = inner_splitted_loop_info->get_outer_splitted_loop_info();
            if (utils::one_of(outer, loop_info_upper, loop_info_lower))
                inner_splitted_loop_info->set_outer_splitted_loop_info(m_map[to]);
        }
    }

    for (auto it = loop_begin_target; it != loop_end_target; ++it) {
        const auto& expr = *it;
        replace_loop_id(expr, from, to);
    }

    remove_loop_info(from);
}

void LoopManager::fuse_loop_ports(std::vector<LoopPort>& output_ports,
                                  std::vector<LoopPort>& input_ports,
                                  size_t loop_id) {
    auto is_loop_id_found = [](const std::vector<size_t>& ids, size_t id) {
        return std::find(ids.cbegin(), ids.cend(), id) != ids.cend();
    };

    std::vector<LoopPort> new_output_ports;
    for (const auto& output_port : output_ports) {
        const auto consumers_inputs = output_port.get_expr_port()->get_connected_ports();

        std::set<LoopPort> mapped_input_ports;
        std::set<ExpressionPtr> outside_consumers;
        for (const auto& consumer_input : consumers_inputs) {
            const auto input_port_it = std::find_if(input_ports.begin(), input_ports.end(),
                                                     [&consumer_input](const LoopPort& port) {
                                                             return *port.get_expr_port().get() == consumer_input;
                                                         });
            if (input_port_it != input_ports.end()) {
                mapped_input_ports.insert(*input_port_it);
                continue;
            }

            const auto& consumer = consumer_input.get_expr();
            const auto& loop_ids = consumer->get_loop_ids();
            if (!is_loop_id_found(loop_ids, loop_id)) {
                outside_consumers.insert(consumer);
            }
        }

        // Remove input ports which are mapped
        auto last_point = input_ports.end();
        for (const auto& mapped_input_port : mapped_input_ports) {
            last_point = std::remove(input_ports.begin(), last_point, mapped_input_port);
        }
        input_ports.resize(input_ports.size() - mapped_input_ports.size());

        // Leave output port if there are consumers outside after fusion
        if (!outside_consumers.empty()) {
            new_output_ports.push_back(output_port);
        }
    }

    output_ports = new_output_ports;
}

void LoopManager::update_loop_ports(const ExpressionPtr& expr) {
    auto update_ports = [&](const ov::snippets::lowered::ExpressionPort& connected_port) {
        const auto is_output = connected_port.get_type() == ExpressionPort::Output;
        // Iterate through all Loops of the connected expression
        for (const auto& loop_id : connected_port.get_expr()->get_loop_ids()) {
            const auto& loop_info = get_loop_info(loop_id);
            // If the connected expression port is not Loop port - nothing to update
            // If the target expression is not from the same Loop - nothing to update
            if (!loop_info->is_loop_port(connected_port) || !is_loop_id_found(expr, loop_id))
                continue;

            std::vector<ExpressionPort> new_ports;
            // Check if some ports of target expression must be Loop port
            const auto target_expr_ports = is_output ? expr->get_output_ports() : expr->get_input_ports();
            for (const auto& port : target_expr_ports) {
                if (utils::should_be_loop_port(port, loop_id))
                    new_ports.push_back(port);
            }
            // Leave the connected expression port as Loop port if needed
            if (utils::should_be_loop_port(connected_port, loop_id))
                new_ports.push_back(connected_port);

            // Nothing should be updated
            if (new_ports.size() == 1 && new_ports.front() == connected_port)
                continue;

            loop_info->replace_with_new_ports(connected_port, new_ports);
        }
    };

    // The case with parent loops: source -> target expr
    for (size_t i = 0; i < expr->get_input_count(); ++i) {
        update_ports(expr->get_input_port_connector(i)->get_source());
    }

    // The case with child loops: target expr -> consumers
    for (size_t i = 0; i < expr->get_output_count(); ++i) {
        const auto& consumers = expr->get_output_port_connector(i)->get_consumers();
        for (const auto& consumer : consumers) {
            update_ports(consumer);
        }
    }
}

void LoopManager::sort_loop_ports(const std::vector<size_t>& loop_ids) {
    for (auto& loop_id : loop_ids)
        get_loop_info(loop_id)->sort_ports();
}

void LoopManager::expression_replacement(LinearIR::constExprIt new_expr_begin, LinearIR::constExprIt new_expr_end, const ExpressionPtr& decomposed_expr,
                                         size_t loop_id, const std::vector<ExpressionPort>& entries, const std::vector<ExpressionPort>& exits) {
    for (auto it = new_expr_begin; it!= new_expr_end; ++it) {
        insert_loop_id(*it, loop_id, true);
    }
    remove_loop_id(decomposed_expr, loop_id);

    auto new_entries = entries;
    auto new_exits = exits;
    if (new_entries.empty() || new_exits.empty()) {
        const auto loop_info = get_loop_info(loop_id);
        get_io_loop_ports(new_expr_begin, new_expr_end, new_entries, new_exits);
    }
    for (size_t i = 0; i < decomposed_expr->get_input_count(); ++i) {
        replace_loop_port(loop_id, decomposed_expr->get_input_port(i), new_entries);
    }
    for (size_t i = 0; i < decomposed_expr->get_output_count(); ++i) {
        replace_loop_port(loop_id, decomposed_expr->get_output_port(i), new_exits);
    }
}

bool LoopManager::reorder_identifiers(const std::map<size_t, size_t>& loop_id_map) {
    OPENVINO_ASSERT(loop_id_map.size() == m_map.size(),
                    "Failed to sort reorder IDs: count of the current IDs in `loop_id_map` must be equal to count of existing loops");
    // If all new IDs are the same as original and they're in the `m_map` - nothing to update
    if (std::all_of(loop_id_map.cbegin(), loop_id_map.cend(),
                   [this](const std::pair<size_t, size_t>& p) { return p.first == p.second && m_map.count(p.first) > 0; })) {
        return false;
    }

    std::map<size_t, LoopInfoPtr> ordered_map;
    for (const auto& p : loop_id_map) {
        const auto& current_id = p.first;
        const auto& target_id = p.second;
        OPENVINO_ASSERT(m_map.count(current_id) > 0, "Failed to reorder loop IDs: the Loop with ID ", current_id, " doesn't exist in the map");
        OPENVINO_ASSERT(ordered_map.count(target_id) == 0, "Failed to reorder loop IDs: there are several Loops with the target ID ", target_id);
        ordered_map[target_id] = get_loop_info(current_id);
    }
    m_map = std::move(ordered_map);
    return true;
}

void LoopManager::insert_loop_id(const ExpressionPtr& expr, size_t new_id, bool before, size_t target_id) {
    OPENVINO_ASSERT(m_map.count(new_id) == 1, "Failed marking expression by Loop ID: the Loop with this ID hasn't registered");
    OPENVINO_ASSERT(!is_loop_id_found(expr, new_id), "Expression cannot have several the same Loop IDs");
    auto loop_ids = expr->get_loop_ids();
    auto insert_it = before ? loop_ids.cbegin() : loop_ids.cend();
    if (target_id != SIZE_MAX) {
        insert_it = std::find(loop_ids.cbegin(), loop_ids.cend(), target_id);
        OPENVINO_ASSERT(insert_it != loop_ids.cend(), "Failed add loop ID: target ID hasn't been found");
    }
    loop_ids.insert(insert_it, new_id);
    expr->set_loop_ids(loop_ids);
}

void LoopManager::insert_loop_ids(const ExpressionPtr& expr, const std::vector<size_t>& new_ids, bool before, size_t target_id) {
    OPENVINO_ASSERT(std::all_of(new_ids.cbegin(), new_ids.cend(), [this](const size_t& id) { return m_map.count(id) == 1; }),
                    "Failed marking expression by Loop ID: the Loop with this ID hasn't registered");
    auto loop_ids = expr->get_loop_ids();
    auto insert_it = before ? loop_ids.cbegin() : loop_ids.cend();
    if (target_id != SIZE_MAX) {
        insert_it = std::find(loop_ids.cbegin(), loop_ids.cend(), target_id);
        OPENVINO_ASSERT(insert_it != loop_ids.cend(), "Failed add loop ID: target ID hasn't been found");
    }
    loop_ids.insert(insert_it, new_ids.cbegin(), new_ids.cend());
    expr->set_loop_ids(loop_ids);
    std::unordered_set<size_t> s(loop_ids.cbegin(), loop_ids.cend());
    OPENVINO_ASSERT(s.size() == loop_ids.size(), "Loop IDs must be unique");
}

void LoopManager::replace_loop_id(const ExpressionPtr& expr, size_t prev_id, size_t new_id) {
    OPENVINO_ASSERT(m_map.count(new_id), "Failed marking expression by Loop ID: the Loop with this ID hasn't registered");
    OPENVINO_ASSERT(!is_loop_id_found(expr, new_id), "Expression cannot have several the same Loop IDs");
    auto loop_ids = expr->get_loop_ids();
    auto it = std::find(loop_ids.begin(), loop_ids.end(), prev_id);
    OPENVINO_ASSERT(it != loop_ids.end(),
                    "Expression doesn't have the Loop with ID " + std::to_string(prev_id));
    (*it) = new_id;
    expr->set_loop_ids(loop_ids);
}

void LoopManager::remove_loop_id(const ExpressionPtr& expr, size_t id) {
    auto loop_ids = expr->get_loop_ids();
    const auto it = std::find(loop_ids.cbegin(), loop_ids.cend(), id);
    OPENVINO_ASSERT(it != loop_ids.cend(), "Expression doesn't have the Loop with ID " + std::to_string(id));
    loop_ids.erase(it);
    expr->set_loop_ids(loop_ids);
}

bool LoopManager::is_loop_id_found(const ExpressionPtr& expr, size_t id) {
    const auto loop_ids = expr->get_loop_ids();
    return std::find(loop_ids.cbegin(), loop_ids.cend(), id) != loop_ids.cend();
}

}// namespace lowered
}// namespace snippets
}// namespace ov
