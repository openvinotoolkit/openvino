// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/lowered/softmax_decomposition.hpp"
#include "snippets/pass/lowered/insert_loops_layout.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"
#include <ngraph/pattern/op/wrap_type.hpp>
#include "openvino/pass/pattern/matcher.hpp"

namespace ngraph {
namespace snippets {
namespace pass {
namespace lowered {
using std::make_shared;
SoftmaxDecomposition::SoftmaxDecomposition(size_t vector_size, int32_t buffer_allocation_rank) :
                        m_vector_size{vector_size},
                        m_buffer_allocation_rank(buffer_allocation_rank) {
}

bool SoftmaxDecomposition::run(LoweredExprIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::SoftmaxDecompositionLowered")
    auto match_load = ngraph::pattern::wrap_type<op::Load>();
    auto match_softmax = ngraph::pattern::wrap_type<op::Softmax>({match_load});
    auto match_store = ngraph::pattern::wrap_type<op::Store>({match_softmax});
    auto matcher = std::make_shared<pattern::Matcher>(match_store, "SoftmaxDecompositionLowered");
    bool modified = false;
    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        const auto& op = (*expr_it)->get_node();
        if (matcher->match(op)) {
            const auto& pm = matcher->get_pattern_map();
            const auto load_node = pm.at(match_load);
            const auto load_expr = linear_ir.get_expr_by_node(load_node);
            const auto input_tds = load_expr->get_inputs();
            const auto output_tds = expr_it->get()->get_outputs();
            linear_ir.erase(std::prev(expr_it));
            linear_ir.erase(std::prev(expr_it));
            expr_it = linear_ir.erase(expr_it);
            linear_ir.get_config();
            // We need an iterator to the inserted element
            auto push_node = [&linear_ir, &expr_it](const std::shared_ptr<Node>& n) {
                return std::make_pair(linear_ir.insert(expr_it, n), n);
            };
            std::vector<std::pair<LoweredExprIR::exprIt, LoweredExprIR::exprIt>> loop_begin_end_offsets;
            // Note: VectorBuffer is a special case, since it should go before the initial Load. So we handle it separately
            const auto& vector_buffer_max = push_node(make_shared<op::VectorBuffer>());

            // Max loop
            const auto& load_max_node = std::make_shared<op::Load>(load_node->get_input_source_output(0), m_vector_size);
            auto loop_begin_offset = linear_ir.insert(expr_it, make_shared<LoweredExpr>(load_max_node, input_tds));
            const auto& max = push_node(make_shared<ov::op::v1::Maximum>(load_max_node, vector_buffer_max.second));

            const auto horizon_max = push_node(make_shared<op::HorizonMax>(max.second));
            // Note: loopEnd will be inserted before HorizonMax
            loop_begin_end_offsets.emplace_back(loop_begin_offset, horizon_max.first);
            const auto broadcast_horizon_max = push_node(make_shared<op::BroadcastMove>(horizon_max.second,
                                                                                           horizon_max.second->get_input_partial_shape(0)));
            const auto vector_buffer_sum = push_node(make_shared<op::VectorBuffer>());

            // Note: A Parameter can currently be connected only to one memory access child (usually Load). This is needed
            // for upstream layout propagation. Here we insert op::Nop to indicate that layout from this Load should not
            // be propagated to a parent Parameter.
            const auto& load_sub_node = std::make_shared<op::Load>(load_node->get_input_source_output(0), m_vector_size);
            loop_begin_offset = linear_ir.insert(expr_it, make_shared<LoweredExpr>(load_sub_node, input_tds));
            const auto sub = push_node(make_shared<ov::op::v1::Subtract>(load_sub_node, broadcast_horizon_max.second));
            const auto exp = push_node(make_shared<ov::op::v0::Exp>(sub.second));
            const auto sum = push_node(make_shared<ov::op::v1::Add>(exp.second, vector_buffer_sum.second));
            const auto store_exp = push_node(make_shared<op::Store>(exp.second, m_vector_size));
            //const auto loop_end_sum = push_node(make_shared<op::LoopEnd>());

            const auto horizon_sum = push_node(make_shared<op::HorizonSum>(sum.second));
            loop_begin_end_offsets.emplace_back(loop_begin_offset, horizon_sum.first);
            // Divide is expensive operation, so we decompose it into 1 / x * y, where 1 / x is executed outside loop
            const auto pow = push_node(make_shared<op::PowerStatic>(horizon_sum.second, -1.f));
            const auto broadcast_pow = push_node(make_shared<op::BroadcastMove>(pow.second, horizon_sum.second->get_input_partial_shape(0)));
            const auto buffer_exp = push_node(make_shared<op::Buffer>(store_exp.second, m_buffer_allocation_rank));

            //const auto loop_begin_div = push_node(make_shared<op::LoopBegin>());
            const auto load_div = push_node(make_shared<op::Load>(buffer_exp.second, m_vector_size));
            loop_begin_offset = load_div.first;
            const auto mul = push_node(make_shared<ov::op::v1::Multiply>(load_div.second, broadcast_pow.second));
            const auto store_div_node = make_shared<op::Store>(mul.second, m_vector_size);
            linear_ir.insert(expr_it, make_shared<LoweredExpr>(store_div_node, mul.first->get()->get_outputs(), output_tds));
            loop_begin_end_offsets.emplace_back(loop_begin_offset, expr_it);
            //const auto loop_end_div = push_node(make_shared<op::LoopEnd>());

            /* =========================================== */

            /* ============= Runtime Info ================ */

            // For tail loop we should fill input of Max by float min and
            // input of Sum by zero to avoid math incorrect calculations
            max.second->input(0).get_rt_info()["set_fill"] = uint32_t(0xff7fffff);
            sum.second->input(0).get_rt_info()["set_fill"] = uint32_t(0x00000000);
            for (const auto& begin_end : loop_begin_end_offsets) {
                InsertLoopsLayout::inject_loops(begin_end.first, begin_end.second, linear_ir, 1, m_vector_size);
                if (auto loop_end = as_type_ptr<op::LoopEnd>(std::prev(begin_end.second)->get()->get_node()))
                    // Note: it doesn't matter here if an outer loop is actually present or not. We need to set
                    // has_outer_loop=true, otherwise finalization_offsets will be ignored by the emitter.
                    // Look at optimize_single_evaluation() for more details.
                    loop_end->has_outer_loop = true;
                else
                    throw ngraph_error("Lowered Softmax decopmposition failed to insert a loop");
            }
            modified = true;
        }
    }
    return modified;
}

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph

