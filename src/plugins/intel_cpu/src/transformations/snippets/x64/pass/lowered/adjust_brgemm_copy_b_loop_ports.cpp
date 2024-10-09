// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"

#include "adjust_brgemm_copy_b_loop_ports.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"
#include "transformations/snippets/x64/pass/lowered/expressions/brgemm_copy_b_buffer_expressions.hpp"

#include "snippets/lowered/loop_info.hpp"
#include "snippets/lowered/loop_manager.hpp"

bool ov::intel_cpu::pass::AdjustBrgemmCopyBLoopPorts::run(snippets::lowered::LinearIR& linear_ir,
                                                          snippets::lowered::LinearIR::constExprIt begin,
                                                          snippets::lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::AdjustBrgemmCopyBLoopPorts")

    bool modified = false;

    for (auto expr_it = begin; expr_it != end; expr_it++) {
        const auto& expr = *expr_it;
        const auto& node = expr->get_node();
        if (!ov::is_type<ov::intel_cpu::BrgemmCopyB>(node))
            continue;
        const auto& loop_ids = expr->get_loop_ids();
        const auto& child_ports = expr->get_output_port(0).get_connected_ports();
        // Note: this pass should be executed before Loop insertion, so there is no LooEnd fake dependency
        OPENVINO_ASSERT(child_ports.size() == 1 &&
                        ov::is_type<RepackedWeightsBufferExpression>(child_ports.begin()->get_expr()),
                        "BrgemmCopyB should have one RepackedWeightsBufferExpression child");
        const auto& grandchild_ports = child_ports.begin()->get_expr()->get_output_port(0).get_connected_ports();
        OPENVINO_ASSERT(grandchild_ports.size() == 1, "BrgemmCopyB is supposed to have one grandchild");
        const auto& target_port = *grandchild_ports.begin();
        const auto& target_loop_ids = target_port.get_expr()->get_loop_ids();
//        size_t i = 0;
//        for (; i < loop_ids.size(); i++)
//            OPENVINO_ASSERT(child_loop_ids[i] == loop_ids[i], "Invalid BrgemmCopyB loop configuration");
        // todo: there could be several mismatched loop ids in case of nested blocked loops
        //  what kind of ptr increments we should set in this case?
        const auto blocked_loop_id = target_loop_ids.back();
        OPENVINO_ASSERT(loop_ids.empty() || blocked_loop_id != loop_ids.back(), "Invalid BrgemmCopyB loop configuration");
        const auto& loop_mngr = linear_ir.get_loop_manager();
        const auto& loop_info = loop_mngr->get_loop_info<snippets::lowered::UnifiedLoopInfo>(blocked_loop_id);

        auto caller = [target_port](snippets::lowered::LoopPort& loop_port, snippets::lowered::UnifiedLoopInfo::LoopPortDesc& loop_desc) {
            const auto& p = *loop_port.expr_port;
            if (p.get_type() != snippets::lowered::ExpressionPort::Input ||
                p != target_port)
                    return;
            loop_desc.ptr_increment *= 2;
        };
        std::cerr << "\n";
        for (auto i : loop_info->get_ptr_increments())
            std::cerr << i << ", ";
        std::cerr << "\n";
        loop_info->iterate_through_infos(caller);
        std::cerr << "\n";
        for (auto i : loop_info->get_ptr_increments())
            std::cerr << i << ", ";
        std::cerr << "\n";
    }

    return modified;
}
