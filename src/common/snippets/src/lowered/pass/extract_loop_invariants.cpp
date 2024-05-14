// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/extract_loop_invariants.hpp"
#include "snippets/utils.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

bool ExtractLoopInvariants::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::ExtractLoopInvariants")
    bool modified = false;
    const auto& loop_manager = linear_ir.get_loop_manager();

    auto remove_last_loop_id = [](const std::shared_ptr<Expression>& expr) {
        auto loop_ids = expr->get_loop_ids();
        loop_ids.pop_back();
        expr->set_loop_ids(loop_ids);
    };

    auto get_stride_after_move_outer = [](const LoopPort& loop_port) {
        const auto& expr_port = loop_port.expr_port;
        const auto& layout = expr_port->get_descriptor_ptr()->get_layout();
        const auto& shape = expr_port->get_descriptor_ptr()->get_shape();

        size_t shape_dim_idx = 0;
        if (expr_port->get_type() == ExpressionPort::Input) {
            shape_dim_idx = utils::get_input_dim_idx(layout, loop_port.dim_idx);
        } else if (expr_port->get_type() == ExpressionPort::Output) {
            shape_dim_idx = utils::get_output_dim_idx(layout, loop_port.dim_idx);
        } else {
            OPENVINO_THROW("Unsupported expression port type!");
        }
        shape_dim_idx--;  // dim_idx of outer loop
        int64_t stride = 1;
        for (size_t i = shape_dim_idx + 1; i < shape.size(); ++i) {
            if (utils::is_dynamic_value(shape[i])) {
                return utils::get_dynamic_value<int64_t>();
            }
            stride *= static_cast<int64_t>(shape[i]);
        }
        return stride;
    };

    // move invariant expr to top(outside) of current loop
    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        const auto& next_expr_it = std::next(expr_it);
        if (next_expr_it == linear_ir.end())
            break;
        const auto& next_expr = *next_expr_it;
        const auto& expr = *expr_it;
        if (next_expr->get_loop_ids().size() < expr->get_loop_ids().size()) {
            // expr is the last expr of inner loop
            const auto& loop_ids = expr->get_loop_ids();
            const auto& inner_loop_id = loop_ids.back();
            bool extracte_complete = false;  // true means all extractable exprs are extracted for this loop
            std::vector<std::shared_ptr<Expression>> extracted_exprs;
            while (!extracte_complete) {
                const auto& inner_loop_info = loop_manager->get_loop_info<UnifiedLoopInfo>(inner_loop_id);
                const auto& inner_loop_input_ports = inner_loop_info->get_input_ports();
                if (inner_loop_input_ports.size() == 0 && inner_loop_info->get_output_ports().size() == 0) {
                    loop_manager->remove_loop_info(inner_loop_id);
                    break;
                }
                bool extract_applicable = true;
                for (size_t i = 0; i < inner_loop_input_ports.size(); i++) {  // iter loop ports
                    const auto& port_expr = inner_loop_input_ports[i].expr_port->get_expr();
                    if (std::find(extracted_exprs.cbegin(), extracted_exprs.cend(), port_expr) != extracted_exprs.cend())
                        continue;
                    const auto& expr_input_ports = port_expr->get_input_ports();
                    extract_applicable = true;
                    if (expr_input_ports.size() == 0)
                        extract_applicable = false;
                    for (size_t i = 0; i < expr_input_ports.size(); ++i) {  // iter expr ports
                        const auto& parent = port_expr->get_input_port_connector(i)->get_source().get_expr();
                        bool parent_scalar_with_single_consumer = ov::is_type<snippets::op::Scalar>(parent->get_node()) &&
                            parent->get_output_port_connector(0)->get_consumers().size() == 1;

                        const auto& loop_port = inner_loop_info->find_loop_port(expr_input_ports[i]);
                        if (loop_port == inner_loop_input_ports.end() && !parent_scalar_with_single_consumer) {
                            // expr input port is not a loop input port
                            extract_applicable = false;
                            break;
                        }
                        if (loop_port != inner_loop_input_ports.end()) {
                            if (get_stride_after_move_outer(*loop_port) != 1) {
                                // after move to outside, stride is not 1, then should not move.
                                extract_applicable = false;
                                break;
                            }
                        }
                    }
                    if (extract_applicable) {
                        // extract
                        extracted_exprs.push_back(port_expr);
                        LinearIR::constExprIt inner_loop_begin_pos, inner_loop_end_pos;
                        std::tie(inner_loop_begin_pos, inner_loop_end_pos) = loop_manager->get_loop_bounds(linear_ir, inner_loop_id);
                        for (size_t i = 0; i < port_expr->get_input_count(); ++i) {
                            auto parent = port_expr->get_input_port_connector(i)->get_source().get_expr();
                            if (ov::is_type<snippets::op::Scalar>(parent->get_node())) {
                                // update parent scalar loop id
                                remove_last_loop_id(parent);
                                // find iterator for parent scalar
                                auto parent_scalar_iter = std::find(inner_loop_begin_pos, inner_loop_end_pos, parent);
                                // move if it is not first
                                if (*parent_scalar_iter != *inner_loop_begin_pos) {
                                    linear_ir.move(parent_scalar_iter, inner_loop_begin_pos);
                                } else {
                                    inner_loop_begin_pos++;  // keep expr order, scalar is before consumer
                                }
                            }
                        }
                        // update expr loop id
                        remove_last_loop_id(port_expr);
                        // move if it is not the first
                        if (port_expr != *inner_loop_begin_pos) {
                            auto port_expr_iter = std::find(inner_loop_begin_pos, inner_loop_end_pos, port_expr);
                            linear_ir.move(port_expr_iter, inner_loop_begin_pos);
                        } else {
                            inner_loop_begin_pos++;
                        }

                        // delete expr input ports from loop input points, add expr output ports' consumers if consumed in inner loop to loop input ports.
                        std::vector<ExpressionPort> new_loop_input_ports;
                        for (size_t i = 0; i < port_expr->get_output_count(); i++) {
                            const auto& consumers = port_expr->get_output_port_connector(i)->get_consumers();
                            for (const auto& consumer : consumers) {
                                const auto& loop_ids = consumer.get_expr()->get_loop_ids();
                                if (std::find(loop_ids.cbegin(), loop_ids.cend(), inner_loop_id) != loop_ids.cend()) {
                                    new_loop_input_ports.push_back(consumer);
                                }
                            }
                        }
                        inner_loop_info->update_loop_ports(expr_input_ports, new_loop_input_ports, true);
                        // delete expr out ports from loop out ports directly if it's in loop output ports
                        const auto& loop_out_ports = inner_loop_info->get_output_ports();
                        std::vector<ExpressionPort> exp_out_ports;
                        for (size_t i = 0; i < port_expr->get_output_count(); ++i) {
                            const auto& out_port = port_expr->get_output_port(i);
                            if (inner_loop_info->find_loop_port(out_port) != loop_out_ports.end()) {
                                exp_out_ports.push_back(out_port);
                            }
                        }
                        if (!exp_out_ports.empty()) {
                            std::vector<ExpressionPort> new_ports;
                            inner_loop_info->update_loop_ports(exp_out_ports, new_ports, false);
                        }
                        // need sort after update loop ports. There are possibility that all exprs are moved to outter loop.
                        if (!inner_loop_info->get_input_ports().empty() && !inner_loop_info->get_output_ports().empty()) {
                            loop_manager->sort_loop_ports(inner_loop_begin_pos, inner_loop_end_pos, inner_loop_id);
                        }

                        // update outer loopInfo
                        const auto& outer_loop_ids = port_expr->get_loop_ids();
                        if (outer_loop_ids.size() > 0) {
                            const auto& outer_loop_id = outer_loop_ids.back();
                            const auto& outer_loop_info = loop_manager->get_loop_info<UnifiedLoopInfo>(outer_loop_id);

                            loop_manager->update_loop_ports(port_expr);
                            // add expr input port to outer loop input ports if it's a outer loop io port
                            std::vector<ExpressionPort> add_in_ports;
                            for (size_t i = 0; i < port_expr->get_input_count(); ++i) {
                                const auto& in_port = port_expr->get_input_port(i);
                                const auto& parent_expr = in_port.get_connected_ports().begin()->get_expr();
                                const auto& parent_expr_loop_ids = parent_expr->get_loop_ids();
                                if (std::find(parent_expr_loop_ids.cbegin(), parent_expr_loop_ids.cend(), outer_loop_id) == parent_expr_loop_ids.cend()) {
                                    // parent expr is not in outer loop
                                    add_in_ports.push_back(in_port);
                                }
                            }
                            std::vector<ExpressionPort> removed_ports;
                            outer_loop_info->update_loop_ports(removed_ports, add_in_ports, true);
                            // add expr out port to outer loop out port if it's a outer loop io port
                            std::vector<ExpressionPort> add_out_ports;
                            for (size_t i = 0; i < port_expr->get_output_count(); ++i) {
                                const auto& out_port = port_expr->get_output_port(i);
                                const auto& consumer_ports = out_port.get_connected_ports();
                                for (const auto& consumer : consumer_ports) {
                                    const auto& consumer_expr = consumer.get_expr();
                                    const auto& loop_ids = consumer_expr->get_loop_ids();
                                    if (std::find(loop_ids.cbegin(), loop_ids.cend(), outer_loop_id) == loop_ids.cend()) {
                                        add_out_ports.push_back(out_port);
                                    }
                                }
                            }
                            outer_loop_info->update_loop_ports(removed_ports, add_out_ports, false);

                            // sort
                            LinearIR::constExprIt outer_loop_begin_pos, outer_loop_end_pos;
                            std::tie(outer_loop_begin_pos, outer_loop_end_pos) = loop_manager->get_loop_bounds(linear_ir, outer_loop_id);
                            loop_manager->sort_loop_ports(outer_loop_begin_pos, outer_loop_end_pos, outer_loop_id);
                        }

                        break; // refreshed loop_input_ports, go while() to start again.
                    }
                }
                // no more extractable expr in this loop.
                if (!extract_applicable)
                    extracte_complete = true;
            }
        }
    }

    return modified;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
