// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"

#include "adjust_brgemm_copy_b_loop_ports.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/snippets/x64/pass/lowered/expressions/brgemm_copy_b_buffer_expressions.hpp"

#include "snippets/lowered/loop_manager.hpp"
namespace ov {
namespace intel_cpu {
using UnifiedLoopInfo = ov::snippets::lowered::UnifiedLoopInfo;

int pass::AdjustBrgemmCopyBLoopPorts::get_affected_port_idx(const std::shared_ptr<UnifiedLoopInfo>& loop_info) {
    bool all_dim_idx_zero = true;
    bool first_port = true;
    bool first_port_incremented = false;
    int idx = 0;
    int i = 0;
    auto caller = [&](snippets::lowered::LoopPort &loop_port,
                      UnifiedLoopInfo::LoopPortDesc &loop_desc) {
        const auto& p = *loop_port.expr_port;
        const auto& src_expr = p.get_port_connector_ptr()->get_source().get_expr();
        if (p.get_type() == snippets::lowered::ExpressionPort::Input &&
            is_type<BrgemmCPU>(p.get_expr()->get_node()) &&
            is_type<RepackedWeightsBufferExpression>(src_expr)) {
            const auto& copy_b_grandparent = src_expr->get_input_port_connector(0)->get_source().get_expr();
            OPENVINO_ASSERT(is_type<BrgemmCopyB>(copy_b_grandparent->get_node()),
                            "RepackedWeightsBufferExpression must have BrgemmCopyB as an input");
            idx = i;
        }
        if (first_port) {
            first_port = false;
            first_port_incremented = loop_port.is_incremented;
        }
        all_dim_idx_zero &= loop_port.dim_idx == 0;
        i++;
    };
    loop_info->iterate_through_infos(caller);

    // todo: do we really need to check first_port is incremented?
    // We need to increment stride only in case of N blocking
    if (!first_port_incremented && all_dim_idx_zero)
        return idx;
    return -1;
}

bool pass::AdjustBrgemmCopyBLoopPorts::run(snippets::lowered::LinearIR& linear_ir,
                                                          snippets::lowered::LinearIR::constExprIt begin,
                                                          snippets::lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(pass::itt::domains::SnippetsTransform, "Snippets::AdjustBrgemmCopyBLoopPorts")

    bool modified = false;

    for (auto expr_it = begin; expr_it != end; expr_it++) {
        const auto& expr = *expr_it;
        const auto& node = expr->get_node();
        if (!is_type<BrgemmCopyB>(node))
            continue;
        const auto& loop_ids = expr->get_loop_ids();
        const auto& child_ports = expr->get_output_port(0).get_connected_ports();
        // Note: this pass should be executed before Loop insertion, so there is no LooEnd fake dependency
        OPENVINO_ASSERT(child_ports.size() == 1 &&
                        is_type<RepackedWeightsBufferExpression>(child_ports.begin()->get_expr()),
                        "BrgemmCopyB should have one RepackedWeightsBufferExpression child");
        const auto& grandchild_ports = child_ports.begin()->get_expr()->get_output_port(0).get_connected_ports();
        OPENVINO_ASSERT(grandchild_ports.size() == 1, "BrgemmCopyB is supposed to have one grandchild");
        const auto& target_port = *grandchild_ports.begin();
        const auto& target_loop_ids = target_port.get_expr()->get_loop_ids();

        // If loop ids match, it means there is no blocking loop
        if (target_loop_ids == loop_ids)
            continue;
        OPENVINO_ASSERT(target_loop_ids.size() > loop_ids.size(), "Invalid BrgemmCopyB loop configuration");
        const auto &loop_mngr = linear_ir.get_loop_manager();
        for (auto i = loop_ids.size(); i < target_loop_ids.size(); i++) {
            const auto &loop_info = loop_mngr->get_loop_info<snippets::lowered::UnifiedLoopInfo>(target_loop_ids[i]);
            auto idx = get_affected_port_idx(loop_info);
            if (idx >= 0) {
                int j = 0;
                auto caller = [&](UnifiedLoopInfo::LoopPortDesc& desc) {
                    if (j == idx) {
//                        desc.ptr_increment *= 2;
//                        desc.finalization_offset *= 2;
                    }
                    j++;
                };
                loop_info->iterate_through_descs(caller);
            }
        }
    }

    return modified;
}
} // namespace intel_cpu
} // namespace ov