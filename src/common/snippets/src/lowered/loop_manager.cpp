// Copyright (C) 2023 Intel Corporation
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
using LoopManager = LinearIR::LoopManager;
using LoopPort = LoopManager::LoopPort;
using LoopInfo = LoopManager::LoopInfo;

std::shared_ptr<LoopPort> LoopPort::clone_with_new_expr(const ExpressionPtr& new_expr) const {
    auto new_loop_port = std::make_shared<LoopPort>(*this);
    new_loop_port->expr_port = expr_port->clone_with_new_expr(new_expr);
    return new_loop_port;
}

LoopInfo::LoopInfo(size_t work_amount, size_t increment, size_t dim_idx,
                                          const std::vector<ExpressionPort>& entries, const std::vector<ExpressionPort>& exits)
    : work_amount(work_amount), increment(increment), dim_idx(dim_idx), outer_splited_loop(false) {
    entry_points.reserve(entries.size());
    exit_points.reserve(exits.size());
    for (const auto& port : entries)
        entry_points.emplace_back(port);
    for (const auto& port : exits)
        exit_points.emplace_back(port);
}

std::shared_ptr<LoopInfo> LoopInfo::clone_with_new_expr(const ExressionMap& expr_map) const {
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
    const auto& new_entry_points = clone_loop_ports(entry_points);
    const auto& new_exit_points = clone_loop_ports(exit_points);

    auto new_loop_info = std::make_shared<LoopInfo>(work_amount, increment, dim_idx, new_entry_points, new_exit_points);
    new_loop_info->outer_splited_loop = outer_splited_loop;

    return new_loop_info;
}

std::shared_ptr<LoopManager> LoopManager::clone_with_new_expr(const ExressionMap& expr_map) const {
    auto new_loop_manager = std::make_shared<LoopManager>();
    for (const auto& id_info : m_map)
        new_loop_manager->m_map.insert({id_info.first, id_info.second->clone_with_new_expr(expr_map)});
    new_loop_manager->next_id = next_id;
    return new_loop_manager;
}

bool operator==(const LinearIR::LoopManager::LoopPort& lhs, const LinearIR::LoopManager::LoopPort& rhs) {
    if (&lhs == &rhs)
        return true;
    return lhs.expr_port == rhs.expr_port && lhs.is_incremented == rhs.is_incremented;
}
bool operator!=(const LinearIR::LoopManager::LoopPort& lhs, const LinearIR::LoopManager::LoopPort& rhs) {
    return !(lhs == rhs);
}
bool operator<(const LinearIR::LoopManager::LoopPort& lhs, const LinearIR::LoopManager::LoopPort& rhs) {
    return (lhs.expr_port < rhs.expr_port) || (lhs.expr_port == rhs.expr_port && (lhs.is_incremented < rhs.is_incremented));
}

size_t LinearIR::LoopManager::add_loop_info(const LoopInfoPtr &loop) {
    const auto index = next_id;
    m_map[index] = loop;
    next_id++;
    return index;
}

void LinearIR::LoopManager::remove_loop_info(size_t index) {
    m_map.erase(index);
}

using LoopInfoPtr = LinearIR::LoopManager::LoopInfoPtr;

const std::map<size_t, LoopInfoPtr> &LinearIR::LoopManager::get_map() const {
    return m_map;
}

LoopInfoPtr LinearIR::LoopManager::get_loop_info(size_t index) const {
    const auto it = m_map.find(index);
    OPENVINO_ASSERT(it != m_map.end(), "LoopInformation hasn't been found!");
    return it->second;
}

std::vector<size_t> LinearIR::LoopManager::get_outer_expr_loops(const ExpressionPtr& expr, size_t loop_id) {
    const auto loop_ids = expr->get_loop_ids();
    const auto it = std::find(loop_ids.cbegin(), loop_ids.cend(), loop_id);
    OPENVINO_ASSERT(it != loop_ids.cend(), "Loop ID hasn't been found");
    return std::vector<size_t>(loop_ids.cbegin(), it);
}

void LinearIR::LoopManager::get_loop_bounds(const LinearIR &linear_ir,
                                            size_t loop_id,
                                            LinearIR::constExprIt &loop_begin_pos,
                                            LinearIR::constExprIt &loop_end_pos,
                                            bool loop_ops_inserted) const {
    const auto loop_info = get_loop_info(loop_id);
    get_loop_bounds(linear_ir, loop_info->entry_points, loop_info->exit_points, loop_begin_pos, loop_end_pos, loop_id, loop_ops_inserted);
}

void LinearIR::LoopManager::get_loop_bounds(const LinearIR &linear_ir,
                                            const std::vector<LoopPort>& entries,
                                            const std::vector<LoopPort>& exits,
                                            LinearIR::constExprIt &loop_begin_pos,
                                            LinearIR::constExprIt &loop_end_pos,
                                            size_t loop_id, bool loop_ops_inserted) {
    OPENVINO_ASSERT(!entries.empty(), "Loop must have entry points");
    OPENVINO_ASSERT(!exits.empty(), "Loop must have entry points");
    const auto& entry_expr = entries.front().expr_port->get_expr();
    loop_begin_pos = linear_ir.find(entry_expr);

    // Some operations in Loop can be before first entry points: Scalars, VectorBuffer.
    // We should iterate by them till the expr is in the corresponding Loop
    auto prev_loop_ids = (*std::prev(loop_begin_pos))->get_loop_ids();
    while (std::find(prev_loop_ids.begin(), prev_loop_ids.end(), loop_id) != prev_loop_ids.end()) {
        loop_begin_pos = std::prev(loop_begin_pos);
        prev_loop_ids = (*std::prev(loop_begin_pos))->get_loop_ids();
    }

    if (loop_ops_inserted) {
        const auto loop_begin = ov::as_type_ptr<op::LoopBegin>((*std::prev(loop_begin_pos))->get_node());
        OPENVINO_ASSERT(loop_begin, "Failed explicit loop bounds getting: LoopBegin has not been found");
        const auto loop_end = loop_begin->get_loop_end();
        OPENVINO_ASSERT(loop_end->get_id() == loop_id, "Failed explicit loop bounds getting: Loop bounds with correct ID have not been found");
        loop_begin_pos = std::prev(loop_begin_pos);
        loop_end_pos = linear_ir.find_after(loop_begin_pos, linear_ir.get_expr_by_node(loop_end));
    } else {
        // At the moment all Loops must have exit points
        const auto& exit_expr = exits.back().expr_port->get_expr();
        loop_end_pos = std::next(linear_ir.find_after(loop_begin_pos, exit_expr));
    }
}

LinearIR::LoopManager::LoopPort LinearIR::LoopManager::get_loop_port_by_expr_port(const ExpressionPort& expr_port, const size_t loop_id) {
    auto get_loop_port = [&](const std::vector<LinearIR::LoopManager::LoopPort>& ports) {
        auto it = std::find_if(ports.cbegin(), ports.cend(), [&](const LinearIR::LoopManager::LoopPort& p) { return *p.expr_port == expr_port; });
        if (it == ports.cend())
            OPENVINO_THROW("Expression has not been found among loop ports. Loop id: " + std::to_string(loop_id));
        return *it;
    };
    const auto& loop_info = get_loop_info(loop_id);
    return expr_port.get_type() == ExpressionPort::Input ? get_loop_port(loop_info->entry_points)
                                                         : get_loop_port(loop_info->exit_points);
}

void LinearIR::LoopManager::get_io_loop_ports(LinearIR::constExprIt loop_begin_pos,
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

void LinearIR::LoopManager::mark_loop(LinearIR::constExprIt loop_begin_pos,
                                      LinearIR::constExprIt loop_end_pos,
                                      size_t loop_depth, size_t vector_size) {
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
        OPENVINO_ASSERT(lhs_value == rhs_value || lhs_value == 1 || rhs_value == 1,
                        "Output shapes of Loop must be broadcastable!");
        *(lhs.rbegin() + index) = std::max(lhs_value, rhs_value);
    };

    auto is_outside_loop = [](const std::vector<size_t>& subtensor) {
        return std::all_of(subtensor.begin(), subtensor.end(), [](size_t lhs) { return lhs == PortDescriptor::ServiceDimensions::FULL_DIM; });
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

        const size_t resizing_value = is_outside_loop(subtensor) ? PortDescriptor::ServiceDimensions::FULL_DIM : 1;
        while (subtensor.size() < loop_depth)
            subtensor.insert(subtensor.begin(), resizing_value);
        if (loop_subtensor.empty())
            loop_subtensor = subtensor;

        OPENVINO_ASSERT(std::equal(loop_subtensor.crbegin(), loop_subtensor.crbegin() + loop_depth, subtensor.crbegin()),
                        "Incorrect scheduling parameters for loop");

        for (size_t dim_idx = 0; dim_idx < loop_depth; ++dim_idx) {
            if (*(subtensor.rbegin() + dim_idx) != PortDescriptor::ServiceDimensions::FULL_DIM) {
                broadcast(loop_tensor, shape, dim_idx);
            }
        }
    }

    for (size_t dim_idx = 0; dim_idx < loop_depth; ++dim_idx) {
        if (*(loop_subtensor.rbegin() + dim_idx) == PortDescriptor::ServiceDimensions::FULL_DIM) {
            continue;
        }

        OPENVINO_ASSERT(dim_idx < loop_tensor.size(), "Incorrect indexes of Loop for markup");
        const auto work_amount =
                loop_tensor.size() > dim_idx ? *(loop_tensor.rbegin() + dim_idx)
                                             : 0;
        const auto work_amount_increment =
                loop_subtensor.size() > dim_idx ? *(loop_subtensor.rbegin() + dim_idx)
                                                : (dim_idx == 0 ? vector_size : 1);
        mark_loop(loop_begin_pos, loop_end_pos, work_amount, work_amount_increment, dim_idx, loop_entry_points, loop_exit_points);
    }
}

void LinearIR::LoopManager::fuse_loops(const LinearIR& linear_ir, size_t loop_id_upper, size_t loop_id_lower, bool fuse_into_upper) {
    LinearIR::constExprIt loop_begin_target, loop_end_target;
    get_loop_bounds(linear_ir, fuse_into_upper ? loop_id_lower : loop_id_upper, loop_begin_target, loop_end_target);
    fuse_loops(loop_begin_target, loop_end_target, loop_id_upper, loop_id_lower, fuse_into_upper);
}

void LinearIR::LoopManager::fuse_loops(LinearIR::constExprIt loop_begin_target, LinearIR::constExprIt loop_end_target,
                                       size_t loop_id_upper, size_t loop_id_lower, bool fuse_into_upper) {
    OPENVINO_ASSERT(m_map.count(loop_id_upper) == 1 && m_map.count(loop_id_lower) == 1,
                    "Failed Loop Fusion: the Loop with the Loop ID isn't existed");

    const auto& loop_info_upper = m_map[loop_id_upper];
    const auto& loop_info_lower = m_map[loop_id_lower];

    auto entry_points_upper = loop_info_upper->entry_points;
    auto exit_points_upper = loop_info_upper->exit_points;
    auto entry_points_lower = loop_info_lower->entry_points;
    auto exit_points_lower = loop_info_lower->exit_points;
    fuse_loop_ports(exit_points_upper, entry_points_lower, loop_id_upper);

    std::vector<LoopManager::LoopPort> new_entries = entry_points_upper;
    new_entries.insert(new_entries.end(), entry_points_lower.begin(), entry_points_lower.end());
    std::vector<LoopManager::LoopPort> new_exits = exit_points_upper;
    new_exits.insert(new_exits.end(), exit_points_lower.begin(), exit_points_lower.end());

    auto& loop_info = fuse_into_upper ? loop_info_upper : loop_info_lower;
    loop_info->entry_points = new_entries;
    loop_info->exit_points = new_exits;

    const auto& from = fuse_into_upper ? loop_id_lower : loop_id_upper;
    const auto& to = fuse_into_upper ? loop_id_upper : loop_id_lower;
    for (auto it = loop_begin_target; it != loop_end_target; ++it) {
        const auto& expr = *it;
        replace_loop_id(expr, from, to);
    }

    remove_loop_info(from);
}

void LinearIR::LoopManager::fuse_loop_ports(std::vector<LinearIR::LoopManager::LoopPort>& exit_points,
                                            std::vector<LinearIR::LoopManager::LoopPort>& entry_points,
                                            size_t loop_id) {
    auto is_loop_id_found = [](const std::vector<size_t>& ids, size_t id) {
        return std::find(ids.cbegin(), ids.cend(), id) != ids.cend();
    };

    std::vector<LinearIR::LoopManager::LoopPort> new_exit_points;
    for (const auto& exit_point : exit_points) {
        const auto consumers_inputs = exit_point.expr_port->get_connected_ports();

        std::set<LinearIR::LoopManager::LoopPort> mapped_entry_points;
        std::set<ExpressionPtr> outside_consumers;
        for (const auto& consumer_input : consumers_inputs) {
            const auto entry_point_it = std::find_if(entry_points.begin(), entry_points.end(),
                                                     [&consumer_input](const LoopManager::LoopPort& point) {
                                                             return *point.expr_port.get() == consumer_input;
                                                         });
            if (entry_point_it != entry_points.end()) {
                mapped_entry_points.insert(*entry_point_it);
                continue;
            }

            const auto& consumer = consumer_input.get_expr();
            const auto loop_ids = consumer->get_loop_ids();
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
void LinearIR::LoopManager::update_loop_port(size_t loop_id, const ExpressionPort& actual_port, const std::vector<ExpressionPort>& target_ports,
                                             bool is_entry) {
    const auto& loop_info = get_loop_info(loop_id);
    auto& ports = is_entry ? loop_info->entry_points : loop_info->exit_points;
    auto port_it = std::find_if(ports.begin(), ports.end(),
                                [&actual_port](const LoopPort& point) { return *point.expr_port.get() == actual_port; });
    // In some cases actual ExpressionPort may not be LoopPort. We shouldn't throw exception here since ExpressionPort is not strong condition as LoopPort
    // For example, not all inner loop ports are ports of outer loops
    if (port_it == ports.end())
        return;

    // to save other parameters except expression port
    std::vector<LoopPort> target_loop_ports;
    target_loop_ports.reserve(target_ports.size());
    for (const auto& p : target_ports)
        target_loop_ports.push_back(*port_it->clone_with_new_expr(p.get_expr()));

    port_it = ports.erase(port_it);
    ports.insert(port_it, target_ports.cbegin(), target_ports.cend());
}

template<>
void LinearIR::LoopManager::update_loop_port(size_t loop_id, const LoopPort& actual_port, const std::vector<LoopPort>& target_ports,
                                             bool is_entry) {
    const auto& loop_info = get_loop_info(loop_id);
    auto& ports = is_entry ? loop_info->entry_points : loop_info->exit_points;
    auto port_it = std::find_if(ports.begin(), ports.end(),
                                [&actual_port](const LoopPort& point) { return point == actual_port; });
    OPENVINO_ASSERT(port_it != ports.end(), "Failed update_loop_port: existing loop ports has not been found");
    port_it = ports.erase(port_it);
    ports.insert(port_it, target_ports.cbegin(), target_ports.cend());
}

void LinearIR::LoopManager::expression_replacement(constExprIt new_expr_begin, constExprIt new_expr_end, const ExpressionPtr& decomposed_expr,
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

void LinearIR::LoopManager::sort_loop_ports(LinearIR::constExprIt& loop_begin_pos, LinearIR::constExprIt& loop_end_pos, size_t loop_id) {
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
    const auto& loop_entries = loop_info->entry_points;
    const auto& loop_exits = loop_info->exit_points;
    std::vector<LoopPort> entries, exits;
    entries.reserve(loop_entries.size());
    exits.reserve(loop_exits.size());
    for (auto it = loop_begin_pos; it != loop_end_pos; ++it) {
        const auto& expr = *it;
        push(loop_entries, entries, expr);
        push(loop_exits, exits, expr);
    }
    loop_info->entry_points = entries;
    loop_info->exit_points = exits;
}

void LinearIR::LoopManager::insert_loop_id(const ExpressionPtr& expr, size_t new_id, bool before, size_t target_id) {
    OPENVINO_ASSERT(m_map.count(new_id) == 1, "Failed marking expression by Loop ID: the Loop with this ID hasn't registered");
    auto& loop_ids = expr->m_loop_ids;
    OPENVINO_ASSERT(std::find(loop_ids.cbegin(), loop_ids.cend(), new_id) == loop_ids.cend(),
                    "Expression cannot have several the same Loop IDs");
    auto insert_it = before ? loop_ids.cbegin() : loop_ids.cend();
    if (target_id != SIZE_MAX) {
        insert_it = std::find(loop_ids.cbegin(), loop_ids.cend(), target_id);
        OPENVINO_ASSERT(insert_it != loop_ids.cend(), "Failed add loop ID: target ID hasn't been found");
    }
    loop_ids.insert(insert_it, new_id);
}

void LinearIR::LoopManager::insert_loop_ids(const ExpressionPtr& expr, const std::vector<size_t>& new_ids, bool before, size_t target_id) {
    OPENVINO_ASSERT(std::all_of(new_ids.cbegin(), new_ids.cend(), [this](const size_t& id) { return m_map.count(id) == 1; }),
                    "Failed marking expression by Loop ID: the Loop with this ID hasn't registered");
    auto& loop_ids = expr->m_loop_ids;
    auto insert_it = before ? loop_ids.cbegin() : loop_ids.cend();
    if (target_id != SIZE_MAX) {
        insert_it = std::find(loop_ids.cbegin(), loop_ids.cend(), target_id);
        OPENVINO_ASSERT(insert_it != loop_ids.cend(), "Failed add loop ID: target ID hasn't been found");
    }
    loop_ids.insert(insert_it, new_ids.cbegin(), new_ids.cend());
    std::unordered_set<size_t> s(loop_ids.cbegin(), loop_ids.cend());
    OPENVINO_ASSERT(s.size() == loop_ids.size(), "Loop IDs must be unique");
}

void LinearIR::LoopManager::replace_loop_id(const ExpressionPtr& expr, size_t prev_id, size_t new_id) {
    OPENVINO_ASSERT(m_map.count(new_id), "Failed marking expression by Loop ID: the Loop with this ID hasn't registered");
    auto& loop_ids = expr->m_loop_ids;
    OPENVINO_ASSERT(std::find(loop_ids.cbegin(), loop_ids.cend(), new_id) == loop_ids.cend(),
                    "Expression already has the Loop with ID " + std::to_string(new_id));
    auto it = std::find(loop_ids.begin(), loop_ids.end(), prev_id);
    OPENVINO_ASSERT(it != loop_ids.end(),
                    "Expression doesn't have the Loop with ID " + std::to_string(prev_id));
    (*it) = new_id;
}

void LinearIR::LoopManager::remove_loop_id(const ExpressionPtr& expr, size_t id) {
    auto& loop_ids = expr->m_loop_ids;
    const auto it = std::find(loop_ids.cbegin(), loop_ids.cend(), id);
    OPENVINO_ASSERT(it != loop_ids.cend(), "Expression doesn't have the Loop with ID " + std::to_string(id));
    loop_ids.erase(it);
}

}// namespace lowered
}// namespace snippets
}// namespace ov
