// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/insert_tail_loop.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

void InsertTailLoop::tail_transformations(LinearIR& linear_ir,
                                          LinearIR::container::const_iterator tail_begin,
                                          LinearIR::container::const_iterator tail_end,
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

    for (auto expr_it = tail_begin; expr_it != tail_end; expr_it++) {
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

bool InsertTailLoop::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::insertTailLoop")
    bool modified = false;
    // *1* solo vector/tail loop + empty outer loop
    //      => skip increments (both counter & ptr) : set evaluate_once flag
    // *2* solo vector/tail loop + non-empty outer loop
    //      => skip counter increments but perform ptr increments : set evaluate_once,
    //         and perform pointer increments through finalization offsets
    // *3* vector loop(s) + one tail loop
    //      => vector as usual, tail depends on outer loop, see *1* and *2*
    auto optimize_single_evaluation = [](const std::shared_ptr<op::LoopEnd>& loop, bool force_ptr_increment = false) {
        if (loop->get_work_amount() < 2 * loop->get_increment()) {
            loop->set_evaluate_once(true);
            if (force_ptr_increment || loop->has_outer_loop) {
                std::vector<int64_t> new_finalization_offsets(loop->get_finalization_offsets());
                const auto& ptr_increments = loop->get_ptr_increments();
                const auto work_amount_incr = static_cast<int64_t>(loop->get_increment());
                for (size_t i = 0; i < new_finalization_offsets.size(); i++) {
                    new_finalization_offsets[i] += ptr_increments[i] * work_amount_incr;
                }
                loop->set_finalization_offsets(new_finalization_offsets);
            }
            return true;
        } else {
            return false;
        }
    };
    auto is_loop_with_buffers = [&linear_ir](const std::shared_ptr<op::LoopEnd>& loop_end) {
        auto is_buffer_input = [](const PortConnectorPtr& input) {
            const auto& parent_expr = input->get_source().get_expr();
            return ov::is_type<op::Buffer>(parent_expr->get_node());
        };
        auto is_buffer_output = [](const PortConnectorPtr& output) {
            const auto child_exprs_inputs = output->get_consumers();
            return std::any_of(child_exprs_inputs.begin(), child_exprs_inputs.end(),
                               [](const ExpressionPort& lp) {return ov::is_type<op::Buffer>(lp.get_expr()->get_node());});
        };

        const auto& loop_end_expr = linear_ir.get_expr_by_node(loop_end);
        const auto inputs = loop_end_expr->get_input_port_connectors();
        const auto in_num = loop_end->get_input_num();
        const auto out_num = loop_end->get_output_num();
        OPENVINO_ASSERT(inputs.size() == (in_num + out_num + 1),
                        std::string("The LoopEnd expression must have the count of inputs is") +
                        std::string("equal to count of input and outputs of Loop plus one for work amount"));
        const std::vector<PortConnectorPtr> loop_ins(inputs.begin(), inputs.begin() + in_num);
        const std::vector<PortConnectorPtr> loop_outs(inputs.begin() + in_num, inputs.begin() + in_num + out_num);
        return std::any_of(loop_ins.begin(), loop_ins.end(), is_buffer_input) ||
               std::any_of(loop_outs.begin(), loop_outs.end(), is_buffer_output);
    };
    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end();) {
        const auto& loop_begin = ov::as_type_ptr<ov::snippets::op::LoopBegin>((*expr_it)->get_node());
        if (!loop_begin) {
            expr_it++;
            continue;
        }

        // ignore outer loops and possible manual scalar loops
        const auto& loop_end = loop_begin->get_loop_end();
        if (loop_end->get_increment() != 1) {
            auto loop_begin_expr_it = expr_it;
            const auto vector_loop_end = loop_end;
            while ((*expr_it)->get_node() != vector_loop_end)
                expr_it++;
            // Note that exp_it points to the element AFTER loop_end
            expr_it++;
            const bool is_there_buffer = is_loop_with_buffers(vector_loop_end);
            const auto work_amount = vector_loop_end->get_work_amount();
            const auto increment = vector_loop_end->get_increment();
            const auto tail_size = work_amount % increment;
            const auto need_tail = tail_size != 0;
            const auto need_vector_loop = work_amount >= increment;
            // Note, that finalization_offsets could be modified inside optimize_single_evaluation,
            // so need to save them here to cover (evaluate_once vector with non-zero finalization_offsets + tail)
            std::vector<int64_t> tail_finalization_offsets = need_tail ? vector_loop_end->get_finalization_offsets()
                                                                       : std::vector<int64_t> {};
            // vector loops are required => Just copy the body, original loop is already a vector one
            if (need_vector_loop) {
                // Note that finalization offsets should be applied after the last iteration.
                // So if there is a tail, then we should apply offsets after it, but not now.
                if (need_tail)
                    vector_loop_end->set_finalization_offsets(
                            std::vector<int64_t>(tail_finalization_offsets.size(), 0));

                // force ptr increments if there is tail
                optimize_single_evaluation(vector_loop_end, need_tail || is_there_buffer);
            }

            // tail is required => transform the body into a tail representation
            // tail loop is fake loop because for tail we should calculate only
            // finalization offsets which are supported by LoopEnd.
            if (need_tail) {
                LinearIR::constExprIt tail_begin;
                LinearIR::constExprIt tail_end;
                if (need_vector_loop) {
                    auto vector_loop_deep_copy = LinearIR::deep_copy_range(loop_begin_expr_it, expr_it);
                    auto is_par_or_res = [](const ExpressionPtr& expr) {
                        return is_type<ov::op::v0::Parameter>(expr->get_node()) ||
                               is_type<ov::op::v0::Result>(expr->get_node());
                    };
                    // Note: It's illegal to insert Parameter or Result to the IR, but they can appear inside vector loop
                    //  So we have to remo them before injecting tail loop into linear_ir
                    auto to_erase = std::remove_if(vector_loop_deep_copy.begin(), vector_loop_deep_copy.end(), is_par_or_res);
                    vector_loop_deep_copy.erase(to_erase, vector_loop_deep_copy.end());
                    tail_begin = linear_ir.insert(expr_it, vector_loop_deep_copy.begin(), vector_loop_deep_copy.end());
                    tail_end = expr_it;
                } else {
                    tail_begin = loop_begin_expr_it;
                    tail_end = expr_it;
                }

                tail_transformations(linear_ir, tail_begin, tail_end, tail_size);
                std::shared_ptr<op::LoopEnd> tail_loop_end =
                        ov::as_type_ptr<op::LoopBegin>((*tail_begin)->get_node())->get_loop_end();
                tail_loop_end->set_finalization_offsets(tail_finalization_offsets);
                tail_loop_end->set_increment(tail_size);
                // ptr increments were set to the old increment, need to update them in accordance with the new one
                tail_loop_end->set_work_amount(tail_size);
                tail_loop_end->has_outer_loop = vector_loop_end->has_outer_loop;

                // Note: despite the fact that the tail loop is always executed once, we still need
                // to keep finalization_offsets to reset Buffer
                optimize_single_evaluation(tail_loop_end, is_there_buffer);
            }
            modified = true;
        } else {
            // if there is a loop, then exprt_it already points to the next statement (after loop end)
            // so we need to increment iterator only if there was no loop
            expr_it++;
        }
    }
    return modified;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov

