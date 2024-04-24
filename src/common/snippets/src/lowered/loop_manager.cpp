// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/loop_manager.hpp"

#include "snippets/lowered/expression.hpp"
#include "snippets/utils.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/core/type.hpp"

#include "snippets/itt.hpp"


namespace ov {
namespace snippets {
namespace lowered {

std::shared_ptr<LoopManager> LoopManager::clone_with_new_expr(const ExpressionMap& expr_map) const {
    auto new_loop_manager = std::make_shared<LoopManager>();
    for (const auto& id_info : m_map)
        new_loop_manager->m_map.insert({id_info.first, id_info.second->clone_with_new_expr(expr_map)});
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

const std::map<size_t, LoopInfoPtr> &LoopManager::get_map() const {
    return m_map;
}

LoopInfoPtr LoopManager::get_loop_info(size_t index) const {
    const auto it = m_map.find(index);
    OPENVINO_ASSERT(it != m_map.end(), "LoopInformation hasn't been found!");
    return it->second;
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
    return get_loop_bounds(linear_ir, loop_id, loop_info->get_entry_points(), loop_info->get_exit_points());
}

std::pair<LinearIR::constExprIt, LinearIR::constExprIt> LoopManager::get_loop_bounds(const LinearIR &linear_ir, size_t loop_id,
                                                                                     const std::vector<LoopPort>& entries,
                                                                                     const std::vector<LoopPort>& exits) {
    OPENVINO_ASSERT(!entries.empty(), "Loop must have entry points");
    OPENVINO_ASSERT(!exits.empty(), "Loop must have entry points");

    const auto& entry_expr = entries.front().expr_port->get_expr();
    auto loop_begin_pos = linear_ir.find(entry_expr);
    // Some operations in Loop can be before first entry points: Scalars, VectorBuffer.
    // We should iterate by them till the expr is in the corresponding Loop
    auto prev_loop_ids = (*std::prev(loop_begin_pos))->get_loop_ids();
    while (std::find(prev_loop_ids.begin(), prev_loop_ids.end(), loop_id) != prev_loop_ids.end()) {
        loop_begin_pos = std::prev(loop_begin_pos);
        prev_loop_ids = (*std::prev(loop_begin_pos))->get_loop_ids();
    }

    const auto& exit_expr = exits.back().expr_port->get_expr();
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
    auto get_loop_port = [&](const std::vector<LoopPort>& ports) {
        auto it = std::find_if(ports.cbegin(), ports.cend(), [&](const LoopPort& p) { return *p.expr_port == expr_port; });
        if (it == ports.cend())
            OPENVINO_THROW("Expression has not been found among loop ports. Loop id: " + std::to_string(loop_id));
        return *it;
    };
    const auto& loop_info = get_loop_info(loop_id);
    return expr_port.get_type() == ExpressionPort::Input ? get_loop_port(loop_info->get_entry_points())
                                                         : get_loop_port(loop_info->get_exit_points());
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
    const auto FULL_DIM = PortDescriptor::ServiceDimensions::FULL_DIM;
    std::vector<ExpressionPort> loop_entry_points, loop_exit_points;
    LoopManager::get_io_loop_ports(loop_begin_pos, loop_end_pos, loop_entry_points, loop_exit_points);

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
        utils::broadcast_merge_dim(*(lhs.rbegin() + index), lhs_value, rhs_value);
    };

    auto is_outside_loop = [&FULL_DIM](const std::vector<size_t>& subtensor) {
        return std::all_of(subtensor.begin(), subtensor.end(), [&FULL_DIM](size_t lhs) { return lhs == FULL_DIM; });
    };

    std::vector<size_t> loop_subtensor;
    std::vector<size_t> loop_tensor(loop_depth, 1);
    for (const auto& exit_point : loop_exit_points) {
        const auto shape = utils::get_preordered_vdims(exit_point);
        auto subtensor = exit_point.get_descriptor_ptr()->get_subtensor();
        if (subtensor.empty()) {
            subtensor.resize(loop_depth, 1);
            subtensor[subtensor.size() - 1] = vector_size;
        }

        const size_t resizing_value = is_outside_loop(subtensor) ? FULL_DIM : 1;
        while (subtensor.size() < loop_depth)
            subtensor.insert(subtensor.begin(), resizing_value);
        if (loop_subtensor.empty())
            loop_subtensor = subtensor;

        OPENVINO_ASSERT(std::equal(loop_subtensor.crbegin(), loop_subtensor.crbegin() + loop_depth, subtensor.crbegin()),
                        "Incorrect scheduling parameters for loop");

        for (size_t dim_idx = 0; dim_idx < loop_depth; ++dim_idx) {
            if (*(subtensor.rbegin() + dim_idx) != FULL_DIM) {
                broadcast(loop_tensor, shape, dim_idx);
            }
        }
    }

    for (size_t dim_idx = 0; dim_idx < loop_depth; ++dim_idx) {
        OPENVINO_ASSERT(dim_idx < loop_subtensor.size(), "Incorrect indexes of Loop for markup");
        const auto& subtensor_value = *(loop_subtensor.rbegin() + dim_idx);
        if (subtensor_value == FULL_DIM) {
            continue;
        }

        OPENVINO_ASSERT(dim_idx < loop_tensor.size(), "Incorrect indexes of Loop for markup");
        const auto work_amount = *(loop_tensor.rbegin() + dim_idx);
        const auto increment = subtensor_value;
        mark_loop(loop_begin_pos, loop_end_pos, work_amount, increment, dim_idx, loop_entry_points, loop_exit_points);
    }
}

size_t LoopManager::replace_with_new_loop(const LinearIR& linear_ir,
                                          LinearIR::constExprIt loop_begin_pos,
                                          LinearIR::constExprIt loop_end_pos,
                                          size_t work_amount,
                                          size_t increment,
                                          const std::vector<LoopPort>& entries,
                                          const std::vector<LoopPort>& exits,
                                          const size_t old_id) {
    const auto is_bound_explicit_loop_begin = ov::is_type<op::LoopBegin>(loop_begin_pos->get()->get_node());
    const auto is_bound_explicit_loop_end = ov::is_type<op::LoopEnd>(std::prev(loop_end_pos)->get()->get_node());
    OPENVINO_ASSERT((is_bound_explicit_loop_begin && is_bound_explicit_loop_end) || (!is_bound_explicit_loop_begin && !is_bound_explicit_loop_end),
                    "Incorrect LoopBounds!");
    const auto explicit_loop_bounds = is_bound_explicit_loop_begin && is_bound_explicit_loop_end;

    const auto loop_id = this->add_loop_info(std::make_shared<LoopInfo>(work_amount, increment, entries, exits));
    const auto begin = explicit_loop_bounds ? std::next(loop_begin_pos) : loop_begin_pos;
    const auto end = explicit_loop_bounds ? std::prev(loop_end_pos) : loop_end_pos;
    for (auto expr_it = begin; expr_it != end; ++expr_it) {
        replace_loop_id(*expr_it, old_id, loop_id);
    }

    // Check that other expression in LinearIR doesn't have the old loop ID - otherwise completely removed from loop manager
    const auto old_loop_bounds = get_loop_bounds(linear_ir, old_id);
    // If new bounds are equal to old loop bounds, this means that old Loop is removed totally from LIR
    // In this case old loop info must be completely removed from loop manager
    if (loop_begin_pos == old_loop_bounds.first && loop_end_pos == old_loop_bounds.second) {
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
    OPENVINO_ASSERT(m_map.count(loop_id_upper) == 1 && m_map.count(loop_id_lower) == 1,
                    "Failed Loop Fusion: the Loop with the Loop ID isn't existed");

    const auto& loop_info_upper = m_map[loop_id_upper];
    const auto& loop_info_lower = m_map[loop_id_lower];

    auto entry_points_upper = loop_info_upper->get_entry_points();
    auto exit_points_upper = loop_info_upper->get_exit_points();
    auto entry_points_lower = loop_info_lower->get_entry_points();
    auto exit_points_lower = loop_info_lower->get_exit_points();
    fuse_loop_ports(exit_points_upper, entry_points_lower, loop_id_upper);

    std::vector<LoopPort> new_entries = entry_points_upper;
    new_entries.insert(new_entries.end(), entry_points_lower.begin(), entry_points_lower.end());
    std::vector<LoopPort> new_exits = exit_points_upper;
    new_exits.insert(new_exits.end(), exit_points_lower.begin(), exit_points_lower.end());

    auto& loop_info = fuse_into_upper ? loop_info_upper : loop_info_lower;
    loop_info->set_entry_points(new_entries);
    loop_info->set_exit_points(new_exits);

    loop_info->set_handlers(SpecificIterationHandlers::merge_handlers(loop_info_upper->get_handlers(), loop_info_lower->get_handlers()));
    // Since fusion can be called for broadcastable loops (one of the loops has work_amount = increment = 1),
    // maximum value is set to the fused loop
    loop_info->set_work_amount(std::max(loop_info_upper->get_work_amount(), loop_info_lower->get_work_amount()));
    loop_info->set_increment(std::max(loop_info_upper->get_increment(), loop_info_lower->get_increment()));

    const auto& from = fuse_into_upper ? loop_id_lower : loop_id_upper;
    const auto& to = fuse_into_upper ? loop_id_upper : loop_id_lower;
    for (auto it = loop_begin_target; it != loop_end_target; ++it) {
        const auto& expr = *it;
        replace_loop_id(expr, from, to);
    }

    remove_loop_info(from);
}

void LoopManager::fuse_loop_ports(std::vector<LoopPort>& exit_points,
                                  std::vector<LoopPort>& entry_points,
                                  size_t loop_id) {
    auto is_loop_id_found = [](const std::vector<size_t>& ids, size_t id) {
        return std::find(ids.cbegin(), ids.cend(), id) != ids.cend();
    };

    std::vector<LoopPort> new_exit_points;
    for (const auto& exit_point : exit_points) {
        const auto consumers_inputs = exit_point.expr_port->get_connected_ports();

        std::set<LoopPort> mapped_entry_points;
        std::set<ExpressionPtr> outside_consumers;
        for (const auto& consumer_input : consumers_inputs) {
            const auto entry_point_it = std::find_if(entry_points.begin(), entry_points.end(),
                                                     [&consumer_input](const LoopPort& point) {
                                                             return *point.expr_port.get() == consumer_input;
                                                         });
            if (entry_point_it != entry_points.end()) {
                mapped_entry_points.insert(*entry_point_it);
                continue;
            }

            const auto& consumer = consumer_input.get_expr();
            const auto& loop_ids = consumer->get_loop_ids();
            if (!is_loop_id_found(loop_ids, loop_id)) {
                outside_consumers.insert(consumer);
            }
        }

        // Remove entry points which are mapped
        auto last_point = entry_points.end();
        for (const auto& mapped_entry_point : mapped_entry_points) {
            last_point = std::remove(entry_points.begin(), last_point, mapped_entry_point);
        }
        entry_points.resize(entry_points.size() - mapped_entry_points.size());

        // Leave exit point if there are consumers outside after fusion
        if (!outside_consumers.empty()) {
            new_exit_points.push_back(exit_point);
        }
    }

    exit_points = new_exit_points;
}

template<>
void LoopManager::update_loop_port(size_t loop_id, const ExpressionPort& actual_port, const std::vector<ExpressionPort>& target_ports, bool is_entry) {
    const auto& loop_info = get_loop_info(loop_id);
    auto ports = is_entry ? loop_info->get_entry_points() : loop_info->get_exit_points();
    auto port_it = std::find_if(ports.begin(), ports.end(),
                                [&actual_port](const LoopPort& point) { return *point.expr_port.get() == actual_port; });
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
    is_entry ? loop_info->set_entry_points(ports) : loop_info->set_exit_points(ports);
}

template<>
void LoopManager::update_loop_port(size_t loop_id, const LoopPort& actual_port, const std::vector<LoopPort>& target_ports, bool is_entry) {
    const auto& loop_info = get_loop_info(loop_id);
    auto ports = is_entry ? loop_info->get_entry_points() : loop_info->get_exit_points();
    auto port_it = std::find_if(ports.begin(), ports.end(),
                                [&actual_port](const LoopPort& point) { return point == actual_port; });
    OPENVINO_ASSERT(port_it != ports.end(), "Failed update_loop_port: existing loop ports has not been found");
    port_it = ports.erase(port_it);
    ports.insert(port_it, target_ports.cbegin(), target_ports.cend());
    is_entry ? loop_info->set_entry_points(ports) : loop_info->set_exit_points(ports);
}

void LoopManager::update_loop_ports(const ExpressionPtr& expr) {
    auto output_ports = expr->get_output_ports();
    for (size_t i = 0; i < expr->get_input_count(); ++i) {
        const auto& source = expr->get_input_port_connector(i)->get_source();
        const auto common_outer_loop_ids = get_common_outer_loops(expr, source.get_expr());
        // The source output port can have several consumers (including the current expr) that can be potential exit points
        // So we should verify on the possible future exit points
        size_t count_of_common_outer_loops = common_outer_loop_ids.size();
        for (const auto& source_consumer : source.get_connected_ports()) {
            if (source_consumer.get_expr() == expr)
                continue;
            count_of_common_outer_loops = std::min(count_of_common_outer_loops, get_common_outer_loops(source.get_expr(), source_consumer.get_expr()).size());
        }
        update_loops_port({common_outer_loop_ids.cbegin(), common_outer_loop_ids.cbegin() + count_of_common_outer_loops}, source, output_ports, false);
        // Save previous port
        if (count_of_common_outer_loops != common_outer_loop_ids.size()) {
            output_ports.insert(output_ports.begin(), source);
            update_loops_port({common_outer_loop_ids.cbegin() + count_of_common_outer_loops, common_outer_loop_ids.cend()}, source, output_ports, false);
        }
    }
    const auto input_ports = expr->get_input_ports();
    for (size_t i = 0; i < expr->get_output_count(); ++i) {
        const auto& consumers = expr->get_output_port_connector(i)->get_consumers();
        for (const auto& consumer : consumers) {
            const auto common_outer_loop_ids = get_common_outer_loops(expr, consumer.get_expr());
            update_loops_port(common_outer_loop_ids, consumer, input_ports, true);
        }
    }
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
        update_loop_port(loop_id, decomposed_expr->get_input_port(i), new_entries);
    }
    for (size_t i = 0; i < decomposed_expr->get_output_count(); ++i) {
        update_loop_port(loop_id, decomposed_expr->get_output_port(i), new_exits, false);
    }
}

void LoopManager::sort_loop_ports(LinearIR::constExprIt& loop_begin_pos, LinearIR::constExprIt& loop_end_pos, size_t loop_id) {
    // The method sorts Loop ports again
    // [113536] Update this logic please, when expression numeration will be implemented
    auto push = [](const std::vector<LoopPort>& ports, std::vector<LoopPort>& sorted_ports, const ExpressionPtr& expr) {
        for (const auto& port : ports) {
            if (port.expr_port->get_expr() == expr) {
                sorted_ports.push_back(port);
            }
        }
    };
    auto loop_info = get_loop_info(loop_id);
    const auto& loop_entries = loop_info->get_entry_points();
    const auto& loop_exits = loop_info->get_exit_points();
    std::vector<LoopPort> entries, exits;
    entries.reserve(loop_entries.size());
    exits.reserve(loop_exits.size());
    for (auto it = loop_begin_pos; it != loop_end_pos; ++it) {
        const auto& expr = *it;
        push(loop_entries, entries, expr);
        push(loop_exits, exits, expr);
    }
    loop_info->set_entry_points(entries);
    loop_info->set_exit_points(exits);
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
