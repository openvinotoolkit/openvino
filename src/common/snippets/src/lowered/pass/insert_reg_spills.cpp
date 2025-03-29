// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/insert_reg_spills.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/op/reg_spill.hpp"
#include "snippets/op/brgemm.hpp"
#include "snippets/itt.hpp"
#include "snippets/utils/utils.hpp"


namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

bool InsertRegSpills::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::InsertRegSpills")

    bool modified = false;
    for (auto it = linear_ir.begin(); it != linear_ir.end(); it++) {
        const auto& expr = *it;
        if (!m_needs_reg_spill(expr))
            continue;
        auto start_it = std::prev(it);
        auto stop_it = std::next(it);
        while (ov::is_type<snippets::op::LoopBegin>(start_it->get()->get_node()) &&
               ov::is_type<snippets::op::LoopEnd>(stop_it->get()->get_node())) {
            start_it--;
            stop_it++;
        }
        // Note: we need to insert immediately before LoopBegin => increment start_it
        start_it++;
        const auto& loop_begin_live = start_it->get()->get_live_regs();
        std::set<Reg> used;
        const auto& reg_info = expr->get_reg_info();
        used.insert(reg_info.first.begin(), reg_info.first.end());
        used.insert(reg_info.second.begin(), reg_info.second.end());
        // Note: before the loop, we need to spill all live regs except for the ones used by the target expression
        std::set<Reg> regs_to_spill;
        std::set_difference(loop_begin_live.begin(), loop_begin_live.end(),
                            used.begin(), used.end(),
                            std::inserter(regs_to_spill, regs_to_spill.begin()));
        // we also need to keep kernel regs alive (actually only abi_param_1 is used in emitters, but save all for consistency)
        for (const auto& r : m_reg_manager.get_kernel_call_regs( snippets::op::Kernel::make_kernel(linear_ir.is_dynamic())))
            regs_to_spill.erase(r);
        if (regs_to_spill.empty())
            continue;
        // All spilled regs are not live anymore => update live_regs for affected expressions
        for (auto affected_it = start_it; affected_it != stop_it; affected_it++) {
            const auto& affected_expr = *affected_it;
            const auto& live_old = affected_expr->get_live_regs();
            std::set<Reg> live_new;
            std::set_difference(live_old.begin(), live_old.end(),
                                regs_to_spill.begin(), regs_to_spill.end(),
                                std::inserter(live_new, live_new.begin()));
            affected_expr->set_live_regs(live_new);
        }

        const auto begin = std::make_shared<op::RegSpillBegin>(regs_to_spill);
        const auto end = std::make_shared<op::RegSpillEnd>(begin);
        const auto loop_ids = start_it->get()->get_loop_ids();
        OPENVINO_ASSERT(loop_ids == std::prev(stop_it)->get()->get_loop_ids(), "Inconsistent loop ids for RegSpill expressions");
        const auto spill_begin_it = linear_ir.insert_node(begin, std::vector<PortConnectorPtr>{}, loop_ids,
                                                          false, start_it, std::vector<std::set<ExpressionPort>>{});
        std::vector<Reg> vregs{regs_to_spill.begin(), regs_to_spill.end()};
        spill_begin_it->get()->set_reg_info({{}, vregs});
        // Note: spill_begin and spill_end do not use any registers, so:
        //  - the regs that are live on entry of spill_begin are the same as for its predecessor (since no regs consumed)
        //  - similarly, live regs for spill_end are the same as for its successor (since no regs produced)
        spill_begin_it->get()->set_live_regs(std::prev(spill_begin_it)->get()->get_live_regs());

        const auto spill_end_it = linear_ir.insert_node(end, spill_begin_it->get()->get_output_port_connectors(), loop_ids,
                                                           false, stop_it, std::vector<std::set<ExpressionPort>>{});
        spill_end_it->get()->set_reg_info({vregs, {}});
        spill_end_it->get()->set_live_regs(std::next(spill_end_it)->get()->get_live_regs());
        modified = true;
    }
    return modified;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov

