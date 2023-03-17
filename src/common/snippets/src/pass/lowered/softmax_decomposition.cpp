// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/lowered/softmax_decomposition.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"
#include <ngraph/pattern/op/wrap_type.hpp>
#include "openvino/pass/pattern/matcher.hpp"
#include "snippets/pass/lowered/loop_markup.hpp"

namespace ngraph {
namespace snippets {
namespace pass {
namespace lowered {

SoftmaxDecomposition::SoftmaxDecomposition(size_t vector_size) : m_vector_size{vector_size} {}

bool SoftmaxDecomposition::run(LoweredExprIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::SoftmaxDecompositionLowered")
    bool modified = false;
    const auto& loop_manager = linear_ir.get_loop_manager();

    auto match_softmax = ngraph::pattern::wrap_type<opset1::Softmax>();
    auto matcher = std::make_shared<pattern::Matcher>(match_softmax, "SoftmaxDecompositionLowered");

    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        const auto& op = (*expr_it)->get_node();
        if (matcher->match(op)) {
            const auto& pm = matcher->get_pattern_map();
            const auto softmax = pm.at(match_softmax);
            const auto softmax_expr = *expr_it;
            const auto input_tds = softmax_expr->get_inputs();
            const auto output_tds = softmax_expr->get_outputs();
            const auto tensor_out = output_tds.front()->get_tensor();
            const auto subtensor_in = input_tds.front()->get_subtensor();
            const auto inner_work_amount = *(tensor_out.rbegin());
            const auto outer_work_amount = *(tensor_out.rbegin() + 1);

            expr_it = linear_ir.erase(expr_it);   // Remove Softmax

            std::vector<LoweredExprPtr> outer_exprs;

            // We need an iterator to the inserted element
            auto push_node = [&linear_ir, &expr_it](const std::shared_ptr<Node>& n) {
                return std::make_pair(linear_ir.insert(expr_it, n), n);
            };

            // Note: VectorBuffer is a special case, since it should go before the initial Load. So we handle it separately
            const auto& vector_buffer_max = push_node(std::make_shared<op::VectorBuffer>());
            outer_exprs.push_back(*vector_buffer_max.first);
            // ReduceMax loop
            const auto& max = push_node(std::make_shared<ov::op::v1::Maximum>(softmax->get_input_source_output(0), vector_buffer_max.second));

            const auto horizon_max = push_node(std::make_shared<op::HorizonMax>(max.second));
            outer_exprs.push_back(*horizon_max.first);

            // Markup of ReduceMax Loop
            loop_manager->mark_loop(linear_ir, max.first, horizon_max.first, 1, inner_work_amount, m_vector_size,
                                  std::vector<LoweredExprPort>{LoweredExprPort::make_input(*max.first, 0),
                                                               LoweredExprPort::make_input(*max.first, 1)},
                                  std::vector<LoweredExprPort>{LoweredExprPort::make_output(*max.first, 0)});

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
            loop_manager->mark_loop(linear_ir, sub.first, horizon_sum.first, 1, inner_work_amount, m_vector_size,
                                  std::vector<LoweredExprPort>{LoweredExprPort::make_input(*sub.first, 0),
                                                               LoweredExprPort::make_input(*sub.first, 1),
                                                               LoweredExprPort::make_input(*sum.first, 1)},
                                  std::vector<LoweredExprPort>{LoweredExprPort::make_output(*exp.first, 0),
                                                               LoweredExprPort::make_output(*sum.first, 0)});

            // Divide is expensive operation, so we decompose it into 1 / x * y, where 1 / x is executed outside loop
            const auto pow = push_node(std::make_shared<op::PowerStatic>(horizon_sum.second, -1.f));
            const auto broadcast_pow = push_node(std::make_shared<op::BroadcastMove>(pow.second, horizon_sum.second->get_input_partial_shape(0)));
            outer_exprs.push_back(*pow.first);
            outer_exprs.push_back(*broadcast_pow.first);

            // Mul (pseudo-Divide loop)
            const auto mul = push_node(std::make_shared<ov::op::v1::Multiply>(exp.second, broadcast_pow.second));

            // Transfer original TensorDescriptors
            linear_ir.replace_input(*max.first, 0, input_tds.front());
            linear_ir.replace_input(*sub.first, 0, input_tds.front());
            linear_ir.replace_output(*mul.first, 0, output_tds.front());

            // Markup of Mul Loop
            loop_manager->mark_loop(linear_ir, mul.first, expr_it, 1, inner_work_amount, m_vector_size,
                                  std::vector<LoweredExprPort>{LoweredExprPort::make_input(*mul.first, 0),
                                                               LoweredExprPort::make_input(*mul.first, 1)},
                                  std::vector<LoweredExprPort>{LoweredExprPort::make_output(*mul.first, 0)});

            // Markup inner loop for outside expression with null loop id
            for (const auto& expr : outer_exprs) {
                expr->set_loop_id(LoweredExpr::LOOP_NULL_ID, 1);
            }

            // Outer Loop
            loop_manager->mark_loop(linear_ir, vector_buffer_max.first, expr_it, 0, outer_work_amount, 1,
                                  std::vector<LoweredExprPort>{LoweredExprPort::make_input(*max.first, 0),
                                                               LoweredExprPort::make_input(*sub.first, 0)},
                                  std::vector<LoweredExprPort>{LoweredExprPort::make_output(*mul.first, 0)});

            /* =========================================== */

            /* ============= Runtime Info ================ */

            // For tail loop we should fill input of Max by float min and
            // input of Sum by zero to avoid math incorrect calculations
            max.second->input(0).get_rt_info()["set_fill"] = uint32_t(0xff7fffff);
            sum.second->input(0).get_rt_info()["set_fill"] = uint32_t(0x00000000);
            modified = true;
        }
    }

    return modified;
}

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph
