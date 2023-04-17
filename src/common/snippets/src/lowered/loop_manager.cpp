// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/loop_manager.hpp"

#include "snippets/lowered/expression.hpp"
#include "snippets/tensor_descriptor.hpp"

#include <openvino/core/graph_util.hpp>
#include <openvino/core/type.hpp>

#include <snippets/itt.hpp>

namespace ngraph {
namespace snippets {
namespace lowered {

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
    get_loop_bounds(linear_ir, loop_info->entry_exprs, loop_info->exit_exprs, loop_begin_pos, loop_end_pos,
                    loop_id);
}

void LinearIR::LoopManager::get_loop_bounds(const LinearIR &linear_ir,
                                            const std::vector<ExpressionPort> &entries,
                                            const std::vector<ExpressionPort> &exits,
                                            LinearIR::constExprIt &loop_begin_pos,
                                            LinearIR::constExprIt &loop_end_pos,
                                            size_t loop_id) {
    OPENVINO_ASSERT(!entries.empty(), "Loop must have entry points");
    OPENVINO_ASSERT(!exits.empty(), "Loop must have entry points");
    loop_begin_pos = std::find(linear_ir.begin(), linear_ir.end(), entries.front().expr);
    OPENVINO_ASSERT(loop_begin_pos != linear_ir.end(), "Loop begin hasn't been found!");

    // Some operations in Loop can be before first entry points: Scalars, VectorBuffer.
    // We should iterate by them till the expr is in the corresponding Loop
    auto prev_loop_ids = (*std::prev(loop_begin_pos))->get_loop_ids();
    while (std::find(prev_loop_ids.begin(), prev_loop_ids.end(), loop_id) != prev_loop_ids.end()) {
        loop_begin_pos = std::prev(loop_begin_pos);
        prev_loop_ids = (*std::prev(loop_begin_pos))->get_loop_ids();
    }

    // At the moment all Loops must have exit points
    loop_end_pos = std::next(std::find(loop_begin_pos, linear_ir.end(), exits.back().expr));
    OPENVINO_ASSERT(loop_end_pos != linear_ir.end(), "Loop end hasn't been found!");
}

void LinearIR::LoopManager::get_io_loop_ports(LinearIR &linear_ir,
                                              LinearIR::constExprIt loop_begin_pos,
                                              LinearIR::constExprIt loop_end_pos,
                                              std::vector<ExpressionPort> &entries,
                                              std::vector<ExpressionPort> &exits) {
    entries.clear();
    exits.clear();
    for (auto expr_it = loop_begin_pos; expr_it != loop_end_pos; ++expr_it) {
        const auto& expr = *expr_it;
        const auto inputs = expr->get_inputs();
        const auto outputs = expr->get_outputs();

        for (size_t in_port = 0; in_port < inputs.size(); ++in_port) {
            const auto in_td = inputs[in_port];
            const auto parent_expr = linear_ir.get_expr_by_output(in_td).expr;
            if (!ov::is_type<ov::op::v0::Constant>(parent_expr->get_node()) &&
                std::find(loop_begin_pos, expr_it, parent_expr) == expr_it) {
                entries.push_back(expr->input_port(in_port));
            }
        }

        for (size_t out_port = 0; out_port < outputs.size(); ++out_port) {
            const auto out_td = outputs[out_port];
            const auto consumer_exprs = linear_ir.get_exprs_by_input(out_td);
            for (const auto& conumer_expr : consumer_exprs) {
                if (std::find(expr_it, loop_end_pos, conumer_expr.expr) == loop_end_pos) {
                    exits.push_back(expr->output_port(out_port));
                    break;
                }
            }
        }
    }
}

void LinearIR::LoopManager::skipped_mark(LinearIR::constExprIt loop_begin_pos,
                                         LinearIR::constExprIt loop_end_pos,
                                         size_t loop_depth) {
    const auto loop_ids = std::vector<size_t>(loop_depth, Expression::LOOP_NULL_ID);
    for (auto& expr_it = loop_begin_pos; expr_it != loop_end_pos; ++expr_it) {
        const auto expr = *expr_it;
        expr->set_loop_ids(loop_ids);
    }
}

void LinearIR::LoopManager::mark_loop(LinearIR &linear_ir,
                                      LinearIR::constExprIt loop_begin_pos,
                                      LinearIR::constExprIt loop_end_pos,
                                      size_t loop_depth, size_t vector_size) {
    std::vector<ExpressionPort> loop_entry_points, loop_exit_points;
    LoopManager::get_io_loop_ports(linear_ir, loop_begin_pos, loop_end_pos, loop_entry_points,
                                          loop_exit_points);

    auto broadcast = [](std::vector<size_t> &lhs, const std::vector<size_t> &rhs) -> void {
        if (rhs == lhs)
            return;
        const auto lhs_size = lhs.size();
        const auto rhs_size = rhs.size();
        const auto size = std::max(lhs_size, rhs_size);
        std::vector<size_t> result(size, 1);
        lhs.resize(size, 1);
        for (size_t i = 0; i < size; ++i) {
            const auto lhs_value = i < lhs_size ? *(lhs.crbegin() + i) : 1;
            const auto rhs_value = i < rhs_size ? *(rhs.crbegin() + i) : 1;
            OPENVINO_ASSERT(lhs_value == rhs_value || lhs_value == 1 || rhs_value == 1,
                            "Output shapes of Loop must be broadcastable!");
            *(lhs.rbegin() + i) = std::max(lhs_value, rhs_value);
        }
    };

    std::vector<size_t> loop_subtensor;
    std::vector<size_t> loop_layout;
    std::vector<size_t> loop_tensor(1, 1);  // Scalar
    for (const auto& exit_point : loop_exit_points) {
        const auto expr = exit_point.expr;
        const auto port = exit_point.port;
        const auto out_td = expr->get_outputs()[port];
        const auto out_tensor = out_td->get_tensor();
        const auto out_layout = out_td->get_layout();
        broadcast(loop_tensor, out_tensor);
        if (loop_layout.empty())
            loop_layout = out_layout;
        OPENVINO_ASSERT(loop_layout == out_layout, "Output layouts of Loop must be the same!");
    }

    for (const auto& entry_point : loop_entry_points) {
        const auto expr = entry_point.expr;
        const auto out_td = expr->get_outputs().front();
        const auto out_subtensor = out_td->get_subtensor();
        if (loop_subtensor.empty())
            loop_subtensor = out_subtensor;
        OPENVINO_ASSERT(loop_subtensor == out_subtensor, "Subtensors of Loop must be the same!");
    }

    for (size_t dim_idx = 0; dim_idx < loop_depth; ++dim_idx) {
        OPENVINO_ASSERT(dim_idx < loop_tensor.size(), "Incorrect indexes of Loop for markup");
        const auto dim = loop_layout.size() >= dim_idx ? *(loop_layout.rbegin() + dim_idx) : 0;
        const auto work_amount = loop_tensor.size() > dim ? loop_tensor[dim] : 0;
        const auto work_amount_increment =
                loop_subtensor.size() > dim_idx ? *(loop_subtensor.rbegin() + dim_idx) :
                dim_idx == 0 ? vector_size : 1;

        mark_loop(linear_ir, loop_begin_pos, loop_end_pos, loop_depth - dim_idx - 1, work_amount,
                  work_amount_increment, loop_entry_points, loop_exit_points);
    }
}

void LinearIR::LoopManager::mark_loop(LinearIR &linear_ir,
                                      LinearIR::constExprIt loop_begin_pos,
                                      LinearIR::constExprIt loop_end_pos,
                                      size_t idx,
                                      size_t work_amount,
                                      size_t work_amount_increment,
                                      const std::vector<ExpressionPort> &entries,
                                      const std::vector<ExpressionPort> &exits) {
    const auto loop_info = std::make_shared<LoopManager::LoopInfo>(
            work_amount, work_amount_increment, entries, exits);
    const auto loop_id = this->add_loop_info(loop_info);
    exprs_marking(loop_begin_pos, loop_end_pos, loop_id, idx);
}

void LinearIR::LoopManager::exprs_marking(LinearIR::constExprIt loop_begin_pos,
                                          LinearIR::constExprIt loop_end_pos,
                                          size_t loop_id, size_t idx) {
    for (auto expr_it = loop_begin_pos; expr_it != loop_end_pos; ++expr_it) {
        expr_it->get()->set_loop_id(loop_id, idx);
    }
}

}// namespace lowered
}// namespace snippets
}// namespace ngraph
