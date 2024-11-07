// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"

#include "adjust_brgemm_copy_b_loop_ports.hpp"
#include "snippets/utils/utils.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/lowered/expressions/buffer_expression.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"

namespace ov {
namespace intel_cpu {

bool pass::AdjustBrgemmCopyBLoopPorts::update_loop_info(const std::shared_ptr<snippets::lowered::UnifiedLoopInfo>& loop_info) {
    OPENVINO_ASSERT(loop_info, "Invalid loop info pointer");
    bool modified = false;
    auto caller = [&](snippets::lowered::LoopPort &loop_port,
                      snippets::lowered::UnifiedLoopInfo::LoopPortDesc &loop_desc) {
        const auto& p = *loop_port.expr_port;
        if (p.get_type() == snippets::lowered::ExpressionPort::Input &&
            p.get_index() == 1) {
            const auto& node = p.get_expr()->get_node();
            if (auto brg = as_type_ptr<BrgemmCPU>(node)) {
                const auto precision = node->get_input_element_type(1);
                /*
                 * The BrgemmCopyB operation repacks the weights in the following way:
                 *  1) VNNI format is applied: KN4k for I8/U8, or KN2k for BF16
                 *  2) Zero padding is applied if N4k < 256 or N2k < 64
                 */
                if (brgemm_utils::with_repacking(brg->get_type()) &&
                    precision != element::f32 && loop_port.is_incremented) {
                    // K blocking loop: account for zero padding
                    if (loop_port.dim_idx == 1) {
                        const auto ptr_incr = loop_desc.ptr_increment;
                        const auto blocked_shape_ptr_inc = brgemm_utils::repacking::compute_LDB(ptr_incr, precision);
                        if (ptr_incr != 0 && ptr_incr != blocked_shape_ptr_inc) {
                            loop_desc.ptr_increment = blocked_shape_ptr_inc;
                            OPENVINO_ASSERT(loop_desc.finalization_offset % ptr_incr == 0,
                                            "Can't rescale finalization offsets");
                            loop_desc.finalization_offset = loop_desc.ptr_increment *
                                                             (loop_desc.finalization_offset / ptr_incr);
                        }
                    // N blocking loop: account for the VNNI format
                    } else if (loop_port.dim_idx == 0) {
                        auto k_blk_size = static_cast<int64_t>(brgemm_utils::compute_vnni_factor(precision));
                        loop_desc.ptr_increment = snippets::utils::dynamic_safe_mul(loop_desc.ptr_increment, k_blk_size);
                        loop_desc.finalization_offset = snippets::utils::dynamic_safe_mul(loop_desc.finalization_offset, k_blk_size);
                    } else {
                        OPENVINO_THROW("Unexpected loop port dimension index in AdjustBrgemmCopyBLoopPorts");
                    }
                    modified = true;
                }
            }
        }
    };
    loop_info->iterate_through_infos(caller);
    return modified;
}

bool pass::AdjustBrgemmCopyBLoopPorts::run(const snippets::lowered::LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::AdjustBrgemmCopyBLoopPorts")

    bool modified = false;

    for (const auto& expr : linear_ir) {
        const auto& node = expr->get_node();
        if (!is_type<BrgemmCopyB>(node))
            continue;
        const auto& repacking_loop_ids = expr->get_loop_ids();
        const auto& child_ports = expr->get_output_port(0).get_connected_ports();
        OPENVINO_ASSERT(child_ports.size() == 1 &&
                        is_type<snippets::lowered::BufferExpression>(child_ports.begin()->get_expr()),
                        "BrgemmCopyB should have one BufferExpression child");
        auto grandchild_ports = child_ports.begin()->get_expr()->get_output_port(0).get_connected_ports();
        for (const auto& target_port : grandchild_ports) {
            const auto& port_node = target_port.get_expr()->get_node();
            if (!is_type<intel_cpu::BrgemmCPU>(port_node)) {
                OPENVINO_ASSERT(is_type<snippets::op::LoopEnd>(port_node),
                                "Invalid grandchild of BrgemmCopyB");
                continue;
            }
            const auto &brgemm_loop_ids = target_port.get_expr()->get_loop_ids();
            // Continue if there is no blocking loop
            if (brgemm_loop_ids.empty() && repacking_loop_ids.empty())
                continue;
            OPENVINO_ASSERT(brgemm_loop_ids.size() > repacking_loop_ids.size(), "Invalid BrgemmCopyB loop configuration");
            const auto &loop_manager = linear_ir.get_loop_manager();
            for (auto i = repacking_loop_ids.size(); i < brgemm_loop_ids.size(); i++) {
                const auto &loop = loop_manager->get_loop_info(brgemm_loop_ids[i]);
                auto uni_loop = ov::as_type_ptr<snippets::lowered::UnifiedLoopInfo>(loop);
                if (!uni_loop)
                    uni_loop = ov::as_type_ptr<snippets::lowered::ExpandedLoopInfo>(loop)->get_unified_loop_info();
                if (!m_affected_loops.count(uni_loop) && update_loop_info(uni_loop)) {
                    m_affected_loops.insert(uni_loop);
                    modified = true;
                }
            }
        }
    }

    return modified;
}
} // namespace intel_cpu
} // namespace ov