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
            const auto& softmax_loop_ids = softmax_expr->get_loop_ids();
            const auto& input_connector = softmax_expr->get_input_port_connector(0);
            const auto& output_connector = softmax_expr->get_output_port_connector(0);
            const auto tensor_out = softmax_expr->get_output_port_descriptor(0)->get_shape();
            const auto inner_work_amount = *(tensor_out.rbegin());

            // Float constant values in byte representation
            const auto float_min_constant = uint32_t(0xff7fffff);
            const auto zero_constant = uint32_t(0x00000000);
            const bool is_dynamic = softmax->is_dynamic();
            // We need an iterator to the inserted element
            auto push_node = [&linear_ir, &expr_it, is_dynamic](const std::shared_ptr<Node>& n) {
                const auto expr = linear_ir.insert(expr_it, n);
                if (is_dynamic)
                    expr->get()->updateShapes();
                return std::make_pair(expr, n);
            };

            // Note: VectorBuffer is a special case, since it should go before the initial Load. So we handle it separately
            const auto& vector_buffer_max = push_node(std::make_shared<op::VectorBuffer>());
            // Init value of vector buffer for ReduceMax is -FLOAT_MIN.
            const auto fill_max = push_node(std::make_shared<op::Fill>(vector_buffer_max.second, 0, float_min_constant));
            // ReduceMax loop
            const auto& max = push_node(std::make_shared<ov::op::v1::Maximum>(softmax->get_input_source_output(0), fill_max.second));

            const auto horizon_max = push_node(std::make_shared<op::HorizonMax>(max.second));

            // Markup of ReduceMax Loop
            loop_manager->mark_loop(max.first, horizon_max.first, inner_work_amount, m_vector_size, 0,
                                    std::vector<ExpressionPort>{(*max.first)->get_input_port(0),
                                                                (*max.first)->get_input_port(1)},
                                    std::vector<ExpressionPort>{(*max.first)->get_output_port(0)});

            const auto vector_buffer_sum = push_node(std::make_shared<op::VectorBuffer>());
            // Init value of vector buffer for ReduceSum is zero.
            const auto fill_sum = push_node(std::make_shared<op::Fill>(vector_buffer_sum.second, 0, zero_constant));

            // Sub + Exp + ReduceSum Loop
            const auto sub = push_node(std::make_shared<ov::op::v1::Subtract>(softmax->get_input_source_output(0), horizon_max.second));
            const auto exp = push_node(std::make_shared<ov::op::v0::Exp>(sub.second));
            const auto sum = push_node(std::make_shared<ov::op::v1::Add>(exp.second, fill_sum.second));

            const auto horizon_sum = push_node(std::make_shared<op::HorizonSum>(sum.second));

            // Markup of ReduceMax Loop
            loop_manager->mark_loop(sub.first, horizon_sum.first, inner_work_amount, m_vector_size, 0,
                                    std::vector<ExpressionPort>{(*sub.first)->get_input_port(0),
                                                                (*sub.first)->get_input_port(1),
                                                                (*sum.first)->get_input_port(1)},
                                    std::vector<ExpressionPort>{(*exp.first)->get_output_port(0),
                                                                (*sum.first)->get_output_port(0)});

            // Divide is expensive operation, so we decompose it into 1 / x * y, where 1 / x is executed outside loop
            const auto pow = push_node(std::make_shared<op::PowerStatic>(horizon_sum.second, -1.f));

            // Mul (pseudo-Divide loop)
            const auto mul = push_node(std::make_shared<ov::op::v1::Multiply>(exp.second, pow.second));

            // Transfer original ExpressionPorts
            replace_input_port_connectors({ max.first->get()->get_input_port(0), sub.first->get()->get_input_port(0) }, input_connector);
            replace_input_port_connectors(output_connector->get_consumers(), (*mul.first)->get_output_port_connector(0));

            // Markup of Mul Loop
            loop_manager->mark_loop(mul.first, expr_it, inner_work_amount, m_vector_size, 0,
                                    std::vector<ExpressionPort>{(*mul.first)->get_input_port(0),
                                                                (*mul.first)->get_input_port(1)},
                                    std::vector<ExpressionPort>{(*mul.first)->get_output_port(0)});

            // Update Loop info for outer loops
            const auto entry_points = std::vector<ExpressionPort>{(*max.first)->get_input_port(0),
                                                                  (*sub.first)->get_input_port(0)};
            const auto exit_points = std::vector<ExpressionPort>{(*mul.first)->get_output_port(0)};
            for (auto loop_id : softmax_loop_ids) {
                loop_manager->expression_replacement(vector_buffer_max.first, expr_it, softmax_expr, loop_id, entry_points, exit_points);
            }

            expr_it = linear_ir.erase(expr_it);   // Remove Softmax

            /* =========================================== */

            /* ============= Runtime Info ================ */

            // For tail loop we should fill input of Max by float min and
            // input of Sum by zero to avoid math incorrect calculations
            // TODO [111383]: It should be covered via general pipeline (for example, via analyze in InsertTailLoop?)
            max.second->input(0).get_rt_info()["set_fill"] = float_min_constant;
            sum.second->input(0).get_rt_info()["set_fill"] = zero_constant;
            modified = true;
        }
    }

    return modified;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
