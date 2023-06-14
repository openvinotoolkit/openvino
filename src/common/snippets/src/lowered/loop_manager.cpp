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

LinearIR::LoopManager::LoopInfo::LoopInfo(size_t work_amount, size_t increment, size_t dim_idx,
                                          const std::vector<ExpressionPort>& entries, const std::vector<ExpressionPort>& exits)
    : work_amount(work_amount), increment(increment), dim_idx(dim_idx) {
    entry_points.reserve(entries.size());
    exit_points.reserve(exits.size());
    for (const auto& port : entries)
        entry_points.emplace_back(port);
    for (const auto& port : exits)
        exit_points.emplace_back(port);
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

void LinearIR::LoopManager::get_loop_bounds(const LinearIR &linear_ir,
                                            size_t loop_id,
                                            LinearIR::constExprIt &loop_begin_pos,
                                            LinearIR::constExprIt &loop_end_pos) const {
    const auto loop_info = get_loop_info(loop_id);
    get_loop_bounds(linear_ir, loop_info->entry_points, loop_info->exit_points, loop_begin_pos, loop_end_pos, loop_id);
}

void LinearIR::LoopManager::get_loop_bounds(const LinearIR &linear_ir,
                                            const std::vector<LoopPort>& entries,
                                            const std::vector<LoopPort>& exits,
                                            LinearIR::constExprIt &loop_begin_pos,
                                            LinearIR::constExprIt &loop_end_pos,
                                            size_t loop_id) {
    OPENVINO_ASSERT(!entries.empty(), "Loop must have entry points");
    OPENVINO_ASSERT(!exits.empty(), "Loop must have entry points");
    const auto& entry_expr = entries.front().expr_port->get_expr();
    loop_begin_pos = std::find(linear_ir.begin(), linear_ir.end(), entry_expr);
    OPENVINO_ASSERT(loop_begin_pos != linear_ir.end(), "Loop begin hasn't been found!");

    // Some operations in Loop can be before first entry points: Scalars, VectorBuffer.
    // We should iterate by them till the expr is in the corresponding Loop
    auto prev_loop_ids = (*std::prev(loop_begin_pos))->get_loop_ids();
    while (std::find(prev_loop_ids.begin(), prev_loop_ids.end(), loop_id) != prev_loop_ids.end()) {
        loop_begin_pos = std::prev(loop_begin_pos);
        prev_loop_ids = (*std::prev(loop_begin_pos))->get_loop_ids();
    }

    // At the moment all Loops must have exit points
    const auto& exit_expr = exits.back().expr_port->get_expr();
    loop_end_pos = std::next(std::find(loop_begin_pos, linear_ir.end(), exit_expr));
    OPENVINO_ASSERT(loop_end_pos != linear_ir.end(), "Loop end hasn't been found!");
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
        const auto& desc = exit_point.get_descriptor_ptr();
        const auto shape = utils::get_reordered_planar_shape(ov::PartialShape(desc->get_shape()), desc->get_layout()).get_shape();
        auto subtensor = desc->get_subtensor();
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

void LinearIR::LoopManager::mark_loop(LinearIR::constExprIt loop_begin_pos,
                                      LinearIR::constExprIt loop_end_pos,
                                      size_t work_amount,
                                      size_t work_amount_increment,
                                      size_t dim_idx,
                                      const std::vector<ExpressionPort>& entries,
                                      const std::vector<ExpressionPort>& exits) {
    const auto loop_info = std::make_shared<LoopManager::LoopInfo>(work_amount, work_amount_increment, dim_idx, entries, exits);
    const auto loop_id = this->add_loop_info(loop_info);
    for (auto expr_it = loop_begin_pos; expr_it != loop_end_pos; ++expr_it) {
        insert_loop_id(*expr_it, loop_id);
    }
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
