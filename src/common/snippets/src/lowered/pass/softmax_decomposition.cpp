// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/softmax_decomposition.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/lowered/pass/mark_loops.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/matcher.hpp"


namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

SoftmaxDecomposition::SoftmaxDecomposition(size_t vector_size) : m_vector_size{vector_size} {}

bool SoftmaxDecomposition::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::SoftmaxDecompositionLowered")
    bool modified = false;
    const auto& loop_manager = linear_ir.get_loop_manager();

    auto match_softmax = ov::pass::pattern::wrap_type<ov::op::v1::Softmax>();
    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(match_softmax, "SoftmaxDecompositionLowered");

    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        const auto& op = (*expr_it)->get_node();
        if (matcher->match(op)) {
            const auto& pm = matcher->get_pattern_map();
            const auto softmax = pm.at(match_softmax);
            const auto softmax_expr = *expr_it;
            const auto softmax_loop_ids = softmax_expr->get_loop_ids();
            const auto& input_connector = softmax_expr->get_input_port_connector(0);
            const auto& output_connector = softmax_expr->get_output_port_connector(0);
            const auto tensor_out = softmax_expr->get_output_port_descriptor(0)->get_shape();
            const auto inner_work_amount = *(tensor_out.rbegin());

            expr_it = linear_ir.erase(expr_it);   // Remove Softmax

            std::vector<ExpressionPtr> outer_exprs;

            // We need an iterator to the inserted element
            auto push_node = [&linear_ir, &expr_it, &softmax_loop_ids](const std::shared_ptr<Node>& n) {
                const auto expr = linear_ir.insert(expr_it, n);
                (*expr)->set_loop_ids(softmax_loop_ids);
                return std::make_pair(expr, n);
            };

            // Note: VectorBuffer is a special case, since it should go before the initial Load. So we handle it separately
            const auto& vector_buffer_max = push_node(std::make_shared<op::VectorBuffer>());
            outer_exprs.push_back(*vector_buffer_max.first);
            // ReduceMax loop
            const auto& max = push_node(std::make_shared<ov::op::v1::Maximum>(softmax->get_input_source_output(0), vector_buffer_max.second));

            const auto horizon_max = push_node(std::make_shared<op::HorizonMax>(max.second));
            outer_exprs.push_back(*horizon_max.first);

            // Markup of ReduceMax Loop
            loop_manager->mark_loop(max.first, horizon_max.first, 1, inner_work_amount, m_vector_size,
                                    std::vector<ExpressionPort>{(*max.first)->get_input_port(0),
                                                                (*max.first)->get_input_port(1)},
                                    std::vector<ExpressionPort>{(*max.first)->get_output_port(0)});

            const auto broadcast_horizon_max = push_node(
                    std::make_shared<op::BroadcastMove>(horizon_max.second, horizon_max.second->get_input_partial_shape(0)));
            const auto vector_buffer_sum = push_node(std::make_shared<op::VectorBuffer>());
            outer_exprs.push_back(*broadcast_horizon_max.first);
            outer_exprs.push_back(*vector_buffer_sum.first);

            // Sub + Exp + ReduceSum Loop
            const auto sub = push_node(std::make_shared<ov::op::v1::Subtract>(softmax->get_input_source_output(0), broadcast_horizon_max.second));
            const auto exp = push_node(std::make_shared<ov::op::v0::Exp>(sub.second));
            const auto sum = push_node(std::make_shared<ov::op::v1::Add>(exp.second, vector_buffer_sum.second));

            const auto horizon_sum = push_node(std::make_shared<op::HorizonSum>(sum.second));
            outer_exprs.push_back(*horizon_sum.first);

            // Markup of ReduceMax Loop
            loop_manager->mark_loop(sub.first, horizon_sum.first, 1, inner_work_amount, m_vector_size,
                                    std::vector<ExpressionPort>{(*sub.first)->get_input_port(0),
                                                                (*sub.first)->get_input_port(1),
                                                                (*sum.first)->get_input_port(1)},
                                    std::vector<ExpressionPort>{(*exp.first)->get_output_port(0),
                                                                (*sum.first)->get_output_port(0)});

            // Divide is expensive operation, so we decompose it into 1 / x * y, where 1 / x is executed outside loop
            const auto pow = push_node(std::make_shared<op::PowerStatic>(horizon_sum.second, -1.f));
            const auto broadcast_pow = push_node(std::make_shared<op::BroadcastMove>(pow.second, horizon_sum.second->get_input_partial_shape(0)));
            outer_exprs.push_back(*pow.first);
            outer_exprs.push_back(*broadcast_pow.first);

            // Mul (pseudo-Divide loop)
            const auto mul = push_node(std::make_shared<ov::op::v1::Multiply>(exp.second, broadcast_pow.second));

            // Transfer original ExpressionPorts
            linear_ir.replace_input((*max.first)->get_input_port(0), input_connector);
            linear_ir.replace_input((*sub.first)->get_input_port(0), input_connector);
            linear_ir.replace_input(output_connector->get_consumers(), (*mul.first)->get_output_port_connector(0));

            // Markup of Mul Loop
            loop_manager->mark_loop(mul.first, expr_it, 1, inner_work_amount, m_vector_size,
                                    std::vector<ExpressionPort>{(*mul.first)->get_input_port(0),
                                                                (*mul.first)->get_input_port(1)},
                                    std::vector<ExpressionPort>{(*mul.first)->get_output_port(0)});

            // Markup inner loop for outside expression with null loop id
            for (const auto& expr : outer_exprs) {
                expr->set_loop_id(Expression::LOOP_NULL_ID, 1);
            }

            auto update_loop_bounds = [&softmax_expr](std::vector<ExpressionPort>& points,
                                                     const std::vector<ExpressionPort>& new_points,
                                                     const LinearIR::LoopManager::LoopInfoPtr& loop_info) {
                auto entry_found = std::find_if(points.begin(), points.end(), [&softmax_expr](const ExpressionPort& desc) {
                    return desc.get_expr() == softmax_expr;
                });
                if (entry_found != points.end()) {
                    entry_found = points.erase(entry_found);
                    points.insert(entry_found, new_points.begin(), new_points.end());
                }
            };

            // Update Loop info for outer loops
            for (auto loop_id : softmax_loop_ids) {
                if (loop_id == Expression::LOOP_NULL_ID)
                    continue;
                const auto loop_info = loop_manager->get_loop_info(loop_id);
                update_loop_bounds(loop_info->entry_exprs, std::vector<ExpressionPort>{(*max.first)->get_input_port(0),
                                                                                       (*sub.first)->get_input_port(0)}, loop_info);
                update_loop_bounds(loop_info->exit_exprs, std::vector<ExpressionPort>{(*mul.first)->get_output_port(0)}, loop_info);
            }

            /* =========================================== */

            /* ============= Runtime Info ================ */

            // For tail loop we should fill input of Max by float min and
            // input of Sum by zero to avoid math incorrect calculations
            // TODO [111383]: It should be covered via general pipeline (for example, via analyze in InsertTailLoop?)
            max.second->input(0).get_rt_info()["set_fill"] = uint32_t(0xff7fffff);
            sum.second->input(0).get_rt_info()["set_fill"] = uint32_t(0x00000000);
            modified = true;
        }
    }

    return modified;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
