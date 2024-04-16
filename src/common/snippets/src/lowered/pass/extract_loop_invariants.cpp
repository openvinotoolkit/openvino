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

    // move invariant expr to top(outside) of current loop
    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        const auto& expr = *expr_it;

        // if there is scalar need to be moved(which has 0 input), move scalar with its consumer, as scalar is before consumer due to MoveScalarToConsumer
        size_t input_num = expr->get_input_count();
        if (input_num <= 0) {
            continue;
        }
        const auto& current_expr_loops = expr->get_loop_ids();
        const auto current_loop_depth = current_expr_loops.size();
        // move from inner loop to outer loop
        if (current_loop_depth != 2) {
            continue;
        }

        size_t outter_loop_id = current_expr_loops[0];
        size_t inner_loop_id = current_expr_loops[1];
        const auto& inner_loop_info = loop_manager->get_loop_info(inner_loop_id);

        bool extract_applicable = true;
        const auto& entry_points = inner_loop_info->get_entry_points();
        for (size_t i = 0; i < input_num; ++i) {
            // last dimension should be 1
            if (expr->get_input_port_descriptor(i)->get_shape().back() != 1) {
                extract_applicable = false;
                break;
            }
            // expr input ports should be loop entry port or scalar(move together), otherwise have inner dependency that prevent move to outter.
            const auto& parent = expr->get_input_port_connector(i)->get_source().get_expr();
            bool parent_scalar_with_single_consumer = ov::is_type<snippets::op::Scalar>(parent->get_node()) &&
                parent->get_output_port_connector(0)->get_consumers().size() == 1;
            if (!loop_manager->is_loop_port(entry_points, expr->get_input_port(i)) && !parent_scalar_with_single_consumer) {
                extract_applicable = false;
                break;
            }
        }

        if (extract_applicable) {
            // move scalar parent first
            const auto& inner_loop_bound = loop_manager->get_loop_bounds(linear_ir, inner_loop_id);
            auto inner_loop_begin = inner_loop_bound.first;
            auto inner_loop_end = inner_loop_bound.second;
            for (size_t i = 0; i < input_num; ++i) {
                auto parent = expr->get_input_port_connector(i)->get_source().get_expr();
                if (ov::is_type<snippets::op::Scalar>(parent->get_node())) {
                    // update parent scalar loop id
                    remove_last_loop_id(parent);
                    // find iterator for parent scalar
                    auto backward_it = std::prev(expr_it);
                    while (*backward_it != parent) {
                        backward_it = std::prev(backward_it);
                    }
                    // move if it is not first
                    if (*backward_it != *inner_loop_begin) {
                        linear_ir.move(backward_it, inner_loop_begin);
                    } else {
                        inner_loop_begin++;  // keep expr order, scalar is before consumer
                    }
                    // scalar has zero input port, not in loop port, no need update loop info. output is consumed by consumer.
                    // loop_manager->update_loop_ports(parent);
                }
            }
            // update expr loop id
            remove_last_loop_id(expr);
            // move if it is not the first
            if (*expr_it != *inner_loop_begin) {
                auto expr_current = expr_it;
                expr_it = std::prev(expr_it);  // save expr_it before move
                linear_ir.move(expr_current, inner_loop_begin);
            } else {
                inner_loop_begin++;
            }

            // update inner loopInfo
            // delete if expr input port is a loop entry point
            auto entry_points = inner_loop_info->get_entry_points();
            for (size_t i = 0; i < expr->get_input_count(); ++i) {
                const auto in_port = expr->get_input_port(i);
                if (loop_manager->is_loop_port(entry_points, in_port)) {
                    std::vector<ExpressionPort> ports = {in_port};
                    loop_manager->delete_loop_ports(inner_loop_id, ports, true);
                }
            }
            // delete if expr output port is a loop exit point
            // if consumers of output port is in inner loop, insert consumer to inner loop entry ports
            const auto& exit_points = inner_loop_info->get_exit_points();
            std::vector<ExpressionPort> insert_entry_ports;
            for (size_t i = 0; i < expr->get_output_count(); ++i) {
                const auto& out_port = expr->get_output_port(i);
                if (loop_manager->is_loop_port(exit_points, out_port)) {
                    // this is to delete expr out port from exit points directly if it's a loop exit point
                    std::vector<ExpressionPort> ports = {out_port};
                    loop_manager->delete_loop_ports(inner_loop_id, ports, false);
                } else {
                    const auto& consumers = expr->get_output_port_connector(i)->get_consumers();
                    for (const auto& consumer : consumers) {
                        const auto& consumer_expr = consumer.get_expr();
                        if (std::find(inner_loop_begin, inner_loop_end, consumer_expr) != inner_loop_end) {
                            insert_entry_ports.push_back(consumer);
                        }
                    }
                    loop_manager->insert_loop_ports(inner_loop_id, insert_entry_ports, true);
                }
            }
            // need sort after insert and delete loop point. There are possibility that all exprs are moved to outter loop
            if (!inner_loop_info->get_exit_points().empty() && !inner_loop_info->get_entry_points().empty()) {
                loop_manager->sort_loop_ports(inner_loop_begin, inner_loop_end, inner_loop_id);
            }

            // update outter loopInfo
            loop_manager->update_loop_ports(expr);
            // add expr input port to outter entry points if it is a outter loop io point
            const auto& outter_loop_bound = loop_manager->get_loop_bounds(linear_ir, outter_loop_id);
            std::vector<ExpressionPort> in_ports;
            for (size_t i = 0; i < expr->get_input_count(); ++i) {
                const auto& in_port = expr->get_input_port(i);
                const auto& parent_expr = in_port.get_connected_ports().begin()->get_expr();
                if (!ov::is_type<ov::op::v0::Constant>(parent_expr->get_node()) &&
                    std::find(outter_loop_bound.first, outter_loop_bound.second, parent_expr) == outter_loop_bound.second) {
                    in_ports.push_back(in_port);
                }
            }
            loop_manager->insert_loop_ports(outter_loop_id, in_ports, true);
            // add expr out port to outter entry points if it is a outter loop io point
            std::vector<ExpressionPort> out_ports;
            for (size_t i = 0; i < expr->get_output_count(); ++i) {
                const auto& out_port = expr->get_output_port(i);
                const auto& consumer_ports = out_port.get_connected_ports();
                for (const auto& consumer : consumer_ports) {
                    const auto& consumer_expr = consumer.get_expr();
                    if (std::find(outter_loop_bound.first, outter_loop_bound.second, consumer_expr) == outter_loop_bound.second) {
                        out_ports.push_back(out_port);
                        break;
                    }
                }
            }
            loop_manager->insert_loop_ports(outter_loop_id, out_ports, false);
            // sort
            auto outter_loop_bound_sort = loop_manager->get_loop_bounds(linear_ir, outter_loop_id);
            loop_manager->sort_loop_ports(outter_loop_bound_sort.first, outter_loop_bound_sort.second, outter_loop_id);

            modified = true;
        }
    }

    // move invariant expr to bottom(outside) of current loop
    for (auto expr_it = linear_ir.rbegin(); expr_it != linear_ir.rend(); expr_it++) {
        const auto& expr = *expr_it;
        const auto& current_expr_loops = expr->get_loop_ids();
        const auto current_loop_depth = current_expr_loops.size();
        // move from inner loop to outter loop
        if (current_loop_depth != 2) {
            continue;
        }

        size_t outter_loop_id = current_expr_loops[0];
        size_t inner_loop_id = current_expr_loops[1];
        const auto& inner_loop_info = loop_manager->get_loop_info(inner_loop_id);

        bool extract_applicable = true;
        // if there is scalar need to be moved(which has 0 input), move scalar with its consumer, as scalar is before consumer due to MoveScalarToConsumer
        size_t input_num = expr->get_input_count();
        if (input_num <= 0) {
            continue;
        }
        for (size_t i = 0; i < input_num; ++i) {
            // last dimension should be 1
            if (expr->get_input_port_descriptor(i)->get_shape().back() != 1) {
                extract_applicable = false;
                break;
            }
        }
        if (extract_applicable) {
            const auto& exit_points = inner_loop_info->get_exit_points();
            size_t output_num = expr->get_output_count();
            for (size_t i = 0; i < output_num; ++i) {
                // expr output port should be loop exit point, and not shared,
                // otherwise output port may be a input(dependency) of other inner expr that prevent move to outter.
                if (!(loop_manager->is_loop_port(exit_points, expr->get_output_port(i)) && expr->get_output_port_connector(i)->get_consumers().size() == 1)) {
                    extract_applicable = false;
                    break;
                }
            }
        }

        if (extract_applicable) {
            const auto& inner_loop_bound = loop_manager->get_loop_bounds(linear_ir, inner_loop_id);
            auto inner_loop_begin = inner_loop_bound.first;
            auto inner_loop_end = inner_loop_bound.second;
            // update expr loop id
            remove_last_loop_id(expr);
            // move expr if it is not the last
            if (*expr_it != *(inner_loop_end--)) {
                auto forward_it = std::prev(expr_it.base());
                expr_it = std::prev(expr_it);               // save expr_it before move
                linear_ir.move(forward_it, inner_loop_end); // move expr before inner_loop_bound.second(next iterator of the last exit loop port)
            }
            // Now inner_loop_end should always expr_it(expr_it is out of inner loop).
            // This resert is needed to keep expr order(parent scalar should insert before expr_it)
            inner_loop_end--;

            // move parent scalar
            for (size_t i = 0; i < input_num; ++i) {
                auto parent = expr->get_input_port_connector(i)->get_source().get_expr();
                if (ov::is_type<snippets::op::Scalar>(parent->get_node())) {
                    remove_last_loop_id(parent);
                    // find iterator for parent scalar
                    auto scalar_it = std::next(expr_it);
                    while (*scalar_it != parent) {
                        scalar_it = std::next(scalar_it);
                    }
                    auto forward_scalar_it = std::prev(scalar_it.base());
                    if (*forward_scalar_it != *(inner_loop_end--)) {
                        linear_ir.move(forward_scalar_it, inner_loop_end);
                    }
                    inner_loop_end--;
                    loop_manager->update_loop_ports(parent);
                }
            }

            // update inner loop info
            // delete if expr input port is loop entry point
            auto entry_points = inner_loop_info->get_entry_points();
            for (size_t i = 0; i < expr->get_input_count(); ++i) {
                const auto in_port = expr->get_input_port(i);
                if (loop_manager->is_loop_port(entry_points, in_port)) {
                    std::vector<ExpressionPort> ports = {in_port};
                    loop_manager->delete_loop_ports(inner_loop_id, ports, true);
                }
            }
            // delete if expr output port is loop exit point
            auto exit_points = inner_loop_info->get_exit_points();
            for (size_t i = 0; i < expr->get_output_count(); ++i) {
                const auto out_port = expr->get_output_port(i);
                if (loop_manager->is_loop_port(exit_points, out_port)) {
                    std::vector<ExpressionPort> ports = {out_port};
                    loop_manager->delete_loop_ports(inner_loop_id, ports, false);
                }
            }
            // insert expr source(not scalar, and parent is in inner loop) to loop exit point
            std::vector<ExpressionPort> insert_exit_ports;
            for (size_t i = 0; i < expr->get_input_count(); ++i) {
                auto source = expr->get_input_port_connector(i)->get_source();
                auto parent = source.get_expr();
                if (!ov::is_type<snippets::op::Scalar>(parent->get_node())) {
                    if (std::find(inner_loop_begin, inner_loop_end, parent) != inner_loop_end) {
                        insert_exit_ports.push_back(source);
                    }
                }
            }
            if (!insert_exit_ports.empty())
                loop_manager->insert_loop_ports(inner_loop_id, insert_exit_ports, false);

            // need sort after insert and delete loop point. There are possibility that all exprs are moved to outter loop
            if (!inner_loop_info->get_exit_points().empty() && !inner_loop_info->get_entry_points().empty()) {
                loop_manager->sort_loop_ports(inner_loop_begin, inner_loop_end, inner_loop_id);
            }

            // update outter loop info.
            loop_manager->update_loop_ports(expr);
            // add expr input port to outter entry points if it is a outter loop io point
            const auto& outter_loop_bound = loop_manager->get_loop_bounds(linear_ir, outter_loop_id);
            std::vector<ExpressionPort> in_ports;
            for (size_t i = 0; i < expr->get_input_count(); ++i) {
                const auto in_port = expr->get_input_port(i);
                const auto parent_expr = in_port.get_connected_ports().begin()->get_expr();
                if (!ov::is_type<ov::op::v0::Constant>(parent_expr->get_node()) &&
                    std::find(outter_loop_bound.first, outter_loop_bound.second, parent_expr) == outter_loop_bound.second) {
                    in_ports.push_back(in_port);
                }
            }
            loop_manager->insert_loop_ports(outter_loop_id, in_ports, true);
            // add expr out port to outter exit points if it is a outter loop io point
            std::vector<ExpressionPort> out_ports;
            for (size_t i = 0; i < expr->get_output_count(); ++i) {
                const auto out_port = expr->get_output_port(i);
                const auto consumer_ports = out_port.get_connected_ports();
                for (const auto& consumer : consumer_ports) {
                    const auto& consumer_expr = consumer.get_expr();
                    if (std::find(outter_loop_bound.first, outter_loop_bound.second, consumer_expr) == outter_loop_bound.second) {
                        out_ports.push_back(out_port);
                        break;
                    }
                }
            }
            loop_manager->insert_loop_ports(outter_loop_id, out_ports, false);
            // sort
            auto outter_loop_bound_sort = loop_manager->get_loop_bounds(linear_ir, outter_loop_id);
            loop_manager->sort_loop_ports(outter_loop_bound_sort.first, outter_loop_bound_sort.second, outter_loop_id);

            modified = true;
        }
    }
    return modified;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
