// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/insert_tail_loop.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/lowered/pass/init_loops.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

std::shared_ptr<op::LoopEnd> InsertTailLoop::create_tail_loop(LinearIR& linear_ir,
                                                              LinearIR::constExprIt vector_begin,
                                                              LinearIR::constExprIt vector_end,
                                                              LinearIR::constExprIt& tail_begin,
                                                              LinearIR::constExprIt& tail_end,
                                                              const std::shared_ptr<op::LoopEnd>& vector_loop_end,
                                                              bool need_vector_loop,
                                                              size_t tail_size,
                                                              const std::vector<int64_t>& tail_finalization_offsets) {
    // tail is required => transform the body into a tail representation
    // tail loop is fake loop because for tail we should calculate only
    // finalization offsets which are supported by LoopEnd.
    if (need_vector_loop) {
        ExressionMap expression_map;
        auto vector_loop_deep_copy = LinearIR::deep_copy_range(vector_begin, vector_end, expression_map);
        tail_begin = linear_ir.insert(vector_end, vector_loop_deep_copy.begin(), vector_loop_deep_copy.end());
        tail_end = vector_end;
    } else {
        tail_begin = vector_begin;
        tail_end = vector_end;
    }

    // We have to check the loop body for any nested loops that work on the same dimension
    // and rescale their work_amount and increment accordingly
    const auto& loop_manager = linear_ir.get_loop_manager();
    const auto& current_loop_Info = loop_manager->get_loop_info(vector_loop_end->get_id());
    if (current_loop_Info->outer_splited_loop) {
        const auto current_dim_idx = current_loop_Info->dim_idx;
        for (auto it = std::next(tail_begin); it != std::prev(tail_end); ++it) {
            const auto& expr = *it;
            const auto inner_loop_end = ov::as_type_ptr<op::LoopEnd>(expr->get_node());
            if (!inner_loop_end)
                continue;
            const auto loop_info = loop_manager->get_loop_info(inner_loop_end->get_id());
            if (loop_info->dim_idx != current_dim_idx)
                continue;
            const auto inner_loop_begin = inner_loop_end->get_loop_begin();
            const auto inner_tail_work_amount = static_cast<int64_t>(inner_loop_end->get_work_amount());
            const auto inner_tail_increment = inner_loop_end->get_increment();
            auto inner_finalization_offsets = inner_loop_end->get_finalization_offsets();
            for (auto& offset : inner_finalization_offsets) {
                offset = offset / inner_tail_work_amount * static_cast<int64_t>(tail_size);
            }
            inner_loop_end->set_work_amount(tail_size);
            inner_loop_end->set_increment(std::min(inner_tail_increment, tail_size));
            inner_loop_end->set_finalization_offsets(inner_finalization_offsets);
            const auto inner_loop_begin_it = std::find(tail_begin, it, linear_ir.get_expr_by_node(inner_loop_begin));
            const auto inner_loop_end_it = std::next(tail_end);
            OPENVINO_ASSERT(inner_loop_begin_it != it, "LoopBegin has not been found!");
            tail_transformations(linear_ir, inner_loop_begin_it, inner_loop_end_it, tail_size);
        }
    }

    tail_transformations(linear_ir, tail_begin, tail_end, tail_size);
    std::shared_ptr<op::LoopEnd> tail_loop_end = ov::as_type_ptr<op::LoopBegin>((*tail_begin)->get_node())->get_loop_end();
    tail_loop_end->set_increment(tail_size);
    // ptr increments were set to the old increment, need to update them in accordance with the new one
    tail_loop_end->set_work_amount(tail_size);
    tail_loop_end->set_finalization_offsets(tail_finalization_offsets);
    tail_loop_end->has_outer_loop = vector_loop_end->has_outer_loop;
    return tail_loop_end;
}

void InsertTailLoop::tail_transformations(LinearIR& linear_ir,
                                          LinearIR::constExprIt tail_begin,
                                          LinearIR::constExprIt tail_end,
                                          const size_t tail_size) {
    const auto& config = linear_ir.get_config();
    auto insertFill = [tail_size](const ov::Input<ov::Node>& input) -> std::shared_ptr<ov::Node> {
        std::shared_ptr<ov::Node> fill = nullptr;
        auto& rt = input.get_rt_info();
        auto fill_rt = rt.find("set_fill");
        if (fill_rt != rt.end()) {
            const auto fill_value = fill_rt->second.as<uint32_t>();
            fill = std::make_shared<ov::snippets::op::Fill>(input.get_source_output(), tail_size, fill_value);
            input.get_node()->set_argument(input.get_index(), fill);
        }
        return fill;
    };

    for (auto expr_it = std::next(tail_begin); expr_it != tail_end; expr_it++) {
        // Skip inner Loops
        const auto loop_begin = ov::as_type_ptr<op::LoopBegin>(expr_it->get()->get_node());
        if (loop_begin) {
            expr_it = linear_ir.find(expr_it, tail_end, linear_ir.get_expr_by_node(loop_begin->get_loop_end()));
            continue;
        }
        // We should fill vector regs by float_min and zero to have
        // correct math calculations for ReduceMax and ReduceSum in scalar case.
        // Note: We find Maximum and Add ops because HorizonMax and HorizonSum are outside Loop,
        //       so they are missed in <tail>
        auto op = (*expr_it)->get_node();
        if (config.m_need_fill_tail_register &&
            (ov::is_type<ov::op::v1::Maximum>(op) ||
             ov::is_type<ov::op::v1::Add>(op))) {
            for (size_t i = 0; i < op->inputs().size(); ++i) {
                if (auto fill = insertFill(op->input(i))) {
                    const auto& input = expr_it->get()->get_input_port_connector(i);
                    const auto consumers = input->get_consumers();
                    auto fill_expr = linear_ir.create_expression(fill, {input});
                    linear_ir.insert(expr_it, fill_expr);
                    linear_ir.replace_input(consumers, fill_expr->get_output_port_connector(0));
                    // in_reg == out_reg since we want to modify vector reg inplace
                    const auto reg = expr_it->get()->get_input_port_descriptor(0)->get_reg();
                    fill_expr->get_input_port_descriptor(0)->set_reg(reg);
                    fill_expr->get_output_port_descriptor(0)->set_reg(reg);
                }
            }
        } else if (const auto memory_access = std::dynamic_pointer_cast<ov::snippets::op::MemoryAccess>(op)) {
            for (const auto p : memory_access->get_memory_access_input_ports()) {
                const auto port = p.first;
                if (memory_access->get_input_count(port) > 1) {
                    memory_access->set_input_count(tail_size, port);
                }
            }
            for (const auto p : memory_access->get_memory_access_output_ports()) {
                const auto port = p.first;
                if (memory_access->get_output_count(port) > 1) {
                    memory_access->set_output_count(tail_size, port);
                }
            }
        }
    }
}

bool InsertTailLoop::optimize_single_evaluation(const std::shared_ptr<op::LoopEnd>& loop) {
    // *1* solo vector/tail loop + empty outer loop
    //      => skip increments (both counter & ptr) : set evaluate_once flag
    // *2* solo vector/tail loop + non-empty outer loop
    //      => skip counter increments but perform ptr increments : set evaluate_once,
    //         and perform pointer increments through finalization offsets
    // *3* vector loop(s) + one tail loop
    //      => vector as usual, tail depends on outer loop, see *1* and *2*
    if (loop->get_work_amount() >= 2 * loop->get_increment())
        return false;

    std::vector<int64_t> new_finalization_offsets(loop->get_finalization_offsets());
    const auto& ptr_increments = loop->get_ptr_increments();
    const auto work_amount_incr = static_cast<int64_t>(loop->get_increment());
    for (size_t i = 0; i < new_finalization_offsets.size(); i++) {
        new_finalization_offsets[i] += ptr_increments[i] * work_amount_incr;
    }
    loop->set_finalization_offsets(new_finalization_offsets);
    loop->set_evaluate_once(true);
    return true;
}

bool InsertTailLoop::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::insertTailLoop")
    const auto& loop_manager = linear_ir.get_loop_manager();
    bool modified = false;

    for (auto expr_it = linear_ir.cbegin(); expr_it != linear_ir.cend(); ++expr_it) {
        const auto& expr = *expr_it;
        const auto node = expr->get_node();
        const auto loop_end = ov::as_type_ptr<op::LoopEnd>(node);
        if (!loop_end)
            continue;

        const auto work_amount = loop_end->get_work_amount();
        const auto increment = loop_end->get_increment();
        const auto loop_info = loop_manager->get_loop_info(loop_end->get_id());
        const auto tail_size = work_amount % increment;
        const auto need_tail = tail_size != 0;
        const auto need_vector_loop = work_amount >= increment;
        // Note, that finalization_offsets could be modified inside optimize_single_evaluation,
        // so need to save them here to cover (evaluate_once vector with non-zero finalization_offsets + tail)
        const auto tail_finalization_offsets = need_tail ? loop_end->get_finalization_offsets() : std::vector<int64_t>{};
        // vector loops are required => Just copy the body, original loop is already a vector one
        if (need_vector_loop) {
            // Note that finalization offsets should be applied after the last iteration.
            // So if there is a tail, then we should apply offsets after it, but not now.
            if (need_tail)
                loop_end->set_finalization_offsets(std::vector<int64_t>(tail_finalization_offsets.size(), 0));

            optimize_single_evaluation(loop_end);
        }

        // tail is required => transform the body into a tail representation
        // tail loop is fake loop because for tail we should calculate only
        // finalization offsets which are supported by LoopEnd.
        if (need_tail) {
            const auto loop_begin = loop_end->get_loop_begin();
            const auto begin_it = linear_ir.find(linear_ir.get_expr_by_node(loop_begin));
            LinearIR::constExprIt tail_begin, tail_end;
            const auto tail_loop_end = create_tail_loop(linear_ir, begin_it, std::next(expr_it), tail_begin, tail_end,
                                                        loop_end, need_vector_loop, tail_size, tail_finalization_offsets);
            optimize_single_evaluation(tail_loop_end);
            // Skip new tail loop. Note: tail_end refs to the next expression after LoopEnd of tail
            expr_it = std::prev(tail_end);
        }
        modified = true;
    }
    return modified;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov

