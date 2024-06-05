// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/extract_loop_invariants.hpp"

#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {
namespace {
void remove_last_loop_id(const std::shared_ptr<Expression>& expr) {
    auto loop_ids = expr->get_loop_ids();
    OPENVINO_ASSERT(!loop_ids.empty(), "Expr loop_ids should not be empty when remove last loop id.");
    loop_ids.pop_back();
    expr->set_loop_ids(loop_ids);
}

size_t get_stride_after_move_outer(const LoopPort& loop_port) {
    const auto& expr_port = loop_port.expr_port;
    const auto& shape = expr_port->get_descriptor_ptr()->get_shape();
    size_t shape_dim_idx = utils::get_dim_idx(*expr_port, loop_port.dim_idx);

    size_t stride = 1;
    for (size_t i = shape_dim_idx; i < shape.size(); ++i) {
        if (utils::is_dynamic_value(shape[i])) {
            return utils::get_dynamic_value<size_t>();
        }
        stride *= shape[i];
    }
    return stride;
}
}  // namespace

bool ExtractLoopInvariants::is_extraction_applicable(const ExpressionPtr& expr,
                                                     const UnifiedLoopInfoPtr& inner_loop_info) {
    const auto& expr_input_ports = expr->get_input_ports();
    const auto& input_port_size = expr_input_ports.size();
    if (input_port_size == 0)
        return false;

    for (size_t i = 0; i < input_port_size; ++i) {  // iter expr ports
        const auto& parent = expr->get_input_port_connector(i)->get_source().get_expr();
        bool parent_scalar_with_single_consumer = ov::is_type<snippets::op::Scalar>(parent->get_node()) &&
                                                  parent->get_output_port_connector(0)->get_consumers().size() == 1;
        const auto& is_loop_port = inner_loop_info->is_loop_port(expr_input_ports[i]);
        // expr input port is not a loop input port, then should not extract
        // expr with single scalar parent could be extracted as well, which is common.
        if (!is_loop_port && !parent_scalar_with_single_consumer) {
            return false;
        }
        if (is_loop_port) {
            // after move to outside, stride in inner loop is not 1, then should not extract.
            const auto& loop_port = inner_loop_info->get_loop_port(expr_input_ports[i]);
            if (get_stride_after_move_outer(loop_port) != 1) {
                return false;
            }
        }
    }
    return true;
}

void ExtractLoopInvariants::extract_expr(const ExpressionPtr& expr,
                                         LinearIR& linear_ir,
                                         LinearIR::constExprIt& inner_loop_begin_pos,
                                         LinearIR::constExprIt& inner_loop_end_pos) {
    // update expr loop id
    remove_last_loop_id(expr);
    // move if it is not the first
    if (expr != *inner_loop_begin_pos) {
        auto port_expr_iter = std::find(inner_loop_begin_pos, inner_loop_end_pos, expr);
        OPENVINO_ASSERT(port_expr_iter != inner_loop_end_pos, "Identified extractable expr is not found in loop.");
        linear_ir.move(port_expr_iter, inner_loop_begin_pos);
    } else {
        inner_loop_begin_pos++;
    }
}

void ExtractLoopInvariants::update_loop_ports(const ExpressionPtr& expr, const LoopManagerPtr& loop_manager, size_t inner_loop_id,
                                              LinearIR::constExprIt& inner_loop_begin_pos, LinearIR::constExprIt& inner_loop_end_pos) {
    const auto& inner_loop_info = loop_manager->get_loop_info<UnifiedLoopInfo>(inner_loop_id);
    // delete expr input ports from loop input points, add expr output ports' consumers if
    // consumed in inner loop to loop input ports.
    std::vector<ExpressionPort> new_loop_input_ports;
    for (size_t i = 0; i < expr->get_output_count(); i++) {
        const auto& consumers = expr->get_output_port_connector(i)->get_consumers();
        for (const auto& consumer : consumers) {
            const auto& loop_ids = consumer.get_expr()->get_loop_ids();
            if (std::find(loop_ids.cbegin(), loop_ids.cend(), inner_loop_id) != loop_ids.cend()) {
                new_loop_input_ports.push_back(consumer);
            }
        }
    }
    const auto& expr_input_ports = expr->get_input_ports();
    inner_loop_info->update_loop_ports(expr_input_ports, new_loop_input_ports, true);

    // delete expr out ports from loop out ports directly if it's in loop output ports
    std::vector<ExpressionPort> exp_out_ports;
    for (size_t i = 0; i < expr->get_output_count(); ++i) {
        const auto& out_port = expr->get_output_port(i);
        if (inner_loop_info->is_loop_port(out_port)) {
            exp_out_ports.push_back(out_port);
        }
    }
    if (!exp_out_ports.empty()) {
        std::vector<ExpressionPort> new_ports;
        inner_loop_info->update_loop_ports(exp_out_ports, new_ports, false);
    }
    // TODO: 142990.
    // Need sort after update loop ports. There are possibility that all exprs are moved to outer loop.
    if (!inner_loop_info->get_input_ports().empty() && !inner_loop_info->get_output_ports().empty()) {
        loop_manager->sort_loop_ports(inner_loop_begin_pos, inner_loop_end_pos, inner_loop_id);
    }
}

std::set<ExpressionPtr> ExtractLoopInvariants::get_potential_extractable_exprs(const std::vector<LoopPort>& loop_in_ports) {
    std::set<ExpressionPtr> expr_set;
    for (size_t i = 0; i < loop_in_ports.size(); ++i) {
        expr_set.insert(loop_in_ports[i].expr_port->get_expr());
    }
    return expr_set;
}

bool ExtractLoopInvariants::extract_from_loop(const size_t& inner_loop_id, LinearIR& linear_ir) {
    const auto& loop_manager = linear_ir.get_loop_manager();
    bool status = false;
    bool extraction_completed = false;  // true means all extractable exprs are extracted from this loop
    const auto& inner_loop_info = loop_manager->get_loop_info<UnifiedLoopInfo>(inner_loop_id);
    while (!extraction_completed) {
        const auto& inner_loop_input_ports = inner_loop_info->get_input_ports();
        const auto& potential_extractable_exprs = get_potential_extractable_exprs(inner_loop_input_ports);
        bool has_been_extraction = false;
        for (const auto& port_expr : potential_extractable_exprs) {
            if (is_extraction_applicable(port_expr, inner_loop_info)) {
                status = true;
                LinearIR::constExprIt inner_loop_begin_pos, inner_loop_end_pos;
                std::tie(inner_loop_begin_pos, inner_loop_end_pos) =
                    loop_manager->get_loop_bounds(linear_ir, inner_loop_id);
                // extract scalar on inputs if there are
                for (size_t i = 0; i < port_expr->get_input_count(); ++i) {
                    auto parent = port_expr->get_input_port_connector(i)->get_source().get_expr();
                    if (ov::is_type<snippets::op::Scalar>(parent->get_node())) {
                        extract_expr(parent, linear_ir, inner_loop_begin_pos, inner_loop_end_pos);
                    }
                }
                extract_expr(port_expr, linear_ir, inner_loop_begin_pos, inner_loop_end_pos);
                update_loop_ports(port_expr, loop_manager, inner_loop_id, inner_loop_begin_pos, inner_loop_end_pos);
                has_been_extraction = true;
                break;  // extracted and refreshed loop_input_ports. break potential_extractable_exprs loop, and go while() to start again.
            }
        }
        if (inner_loop_input_ports.size() == 0 && inner_loop_info->get_output_ports().size() == 0) {
            // become a empty(inner_loop_input_ports is ref) loop after extraction, let remove it from loop_manager
            loop_manager->remove_loop_info(inner_loop_id);
            extraction_completed = true;
            break;
        }
        // no more extractable expr in this loop after for() of iter loop ports, done for this loop.
        if (!has_been_extraction)
            extraction_completed = true;
    }
    return status;
}

bool ExtractLoopInvariants::run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::ExtractLoopInvariants")
    bool modified = false;

    // move invariant expr to top(outside) of current loop
    for (auto expr_it = begin; expr_it != end; expr_it++) {
        const auto& next_expr_it = std::next(expr_it);
        if (next_expr_it == linear_ir.end())
            break;
        const auto& expr = *expr_it;
        const auto& next_expr = *next_expr_it;
        const auto& current_loop_ids = expr->get_loop_ids();
        const auto& next_loop_ids = next_expr->get_loop_ids();
        const auto& current_loop_ids_size = current_loop_ids.size();
        const auto& next_loop_ids_size = next_loop_ids.size();

        if (next_loop_ids_size < current_loop_ids_size) {
            // expr is the last expr of inner loop
            std::vector<size_t> identified_loop_ids(
                current_loop_ids.crbegin(),
                current_loop_ids.crbegin() + (current_loop_ids_size - next_loop_ids_size));

            for (size_t i = 0; i < identified_loop_ids.size(); i++) {
                modified |= extract_from_loop(identified_loop_ids[i], linear_ir);
            }
        }
    }

    return modified;
}

}  // namespace pass
}  // namespace lowered
}  // namespace snippets
}  // namespace ov
