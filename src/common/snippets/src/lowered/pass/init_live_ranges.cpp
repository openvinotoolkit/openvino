// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/init_live_ranges.hpp"
#include "snippets/itt.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/lowered/expressions/buffer_expression.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {
namespace {
// Expressions that don't affect lifetime of registers, e.g. Buffer or RankNormalization
inline bool pass_through_expr(const ExpressionPtr& expr) {
    const auto& node = expr->get_node();
    return op::Subgraph::is_shape_infer_op(node)
#ifdef SNIPPETS_DEBUG_CAPS
            || ov::is_type_any_of<op::PerfCountBeginBase, op::PerfCountEndBase>(node)
#endif
            || ov::is_type<BufferExpression>(expr);
}

} // namespace

bool InitLiveRanges::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::InitLiveRanges")
    std::map<RegType, size_t> reg_counter;

    // Note: map expiring time to register
    std::map<double, std::set<Reg>> regs_to_expire;

    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        const auto& expr = *expr_it;
        const auto op = expr->get_node();
        const double start = expr->get_exec_num();
        std::set<Reg> live_regs;

        if (pass_through_expr(expr)) {
            if (expr_it != linear_ir.begin()) {
                live_regs = std::prev(expr_it)->get()->get_live_regs();
                expr->set_live_regs(std::move(live_regs));
            }
        } else {
            // Remove all regs that expired before start
            regs_to_expire.erase(regs_to_expire.begin(), regs_to_expire.lower_bound(start)); // remove all elements lower than start (not equal)
            for (const auto& time_reg : regs_to_expire)
                live_regs.insert(time_reg.second.begin(), time_reg.second.end());
            expr->set_live_regs(std::move(live_regs));
        }

        // Note that here we continue to process pass_through expressions to define register type if it was not defined
        // in the parent expression (for example, in cases where there are no parent expression for passthrough
        // expressions) and to propagate new registers to the consumers
        for (size_t i = 0; i < expr->get_output_count(); ++i) {
            const auto& out_pd = expr->get_output_port_descriptor(i);
            if (out_pd->get_reg().is_defined())
                continue;
            const auto reg_type = m_reg_manager.get_reg_type(op->output(i));
            const auto& reg = Reg(reg_type, reg_counter[reg_type]++);
            double stop = start;
            // propagate to consumers
            std::stack<PortConnectorPtr> to_visit;
            to_visit.push(expr->get_output_port_connector(i));
            while (!to_visit.empty()) {
                const auto& current = to_visit.top();
                current->get_source().get_descriptor_ptr()->set_reg(reg);
                to_visit.pop();
                for (const auto& consumer : current->get_consumers()) {
                    consumer.get_descriptor_ptr()->set_reg(reg);
                    const auto& consumer_expr = consumer.get_expr();
                    stop = std::max(stop, consumer_expr->get_exec_num());
                    // Note: pass_through expression don't affect registers' life times,
                    // so we should examine their consumers to understand when the register will actually be used
                    if (pass_through_expr(consumer_expr)) {
                        for (const auto& connector : consumer_expr->get_output_port_connectors())
                            to_visit.push(connector);
                    }
                }
            }
            regs_to_expire[stop].insert(reg);
            m_reg_manager.set_live_range(reg, std::make_pair(start, stop));
        }
    }

    return true;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov

