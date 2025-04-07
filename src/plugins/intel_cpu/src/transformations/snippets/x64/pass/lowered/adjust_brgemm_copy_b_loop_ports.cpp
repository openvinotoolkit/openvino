// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "adjust_brgemm_copy_b_loop_ports.hpp"

#include "snippets/itt.hpp"
#include "snippets/lowered/expressions/buffer_expression.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"

namespace ov::intel_cpu {

bool pass::AdjustBrgemmCopyBLoopPorts::update_loop_info(
    const std::shared_ptr<snippets::lowered::UnifiedLoopInfo>& loop_info) {
    OPENVINO_ASSERT(loop_info, "Invalid loop info pointer");
    bool modified = false;
    auto caller = [&](snippets::lowered::LoopPort& loop_port,
                      snippets::lowered::UnifiedLoopInfo::LoopPortDesc& loop_desc) {
        const auto& p = *loop_port.get_expr_port();
        if (p.get_type() == snippets::lowered::ExpressionPort::Input && p.get_index() == 1) {
            const auto& node = p.get_expr()->get_node();
            if (auto brg = as_type_ptr<BrgemmCPU>(node)) {
                const auto precision = node->get_input_element_type(1);
                /*
                 * The BrgemmCopyB operation repacks the weights in the following way:
                 *  1) VNNI format is applied: KN4k for I8/U8, or KN2k for BF16
                 *  2) Zero padding is applied if N4k < 256 or N2k < 64
                 */
                if (brgemm_utils::with_repacking(brg->get_type()) && loop_port.is_incremented()) {
                    // K blocking loop: account for zero padding
                    if (loop_port.get_dim_idx() == 1) {
                        const auto ptr_incr = loop_desc.ptr_increment;
                        const auto blocked_shape_ptr_inc =
                            brgemm_utils::repacking::compute_repacked_n_dim(ptr_incr, precision);
                        if (ptr_incr != 0 && ptr_incr != blocked_shape_ptr_inc) {
                            loop_desc.ptr_increment = blocked_shape_ptr_inc;
                            OPENVINO_ASSERT(loop_desc.finalization_offset % ptr_incr == 0,
                                            "Can't rescale finalization offsets");
                            loop_desc.finalization_offset =
                                loop_desc.ptr_increment * (loop_desc.finalization_offset / ptr_incr);
                        }
                        // N blocking loop: account for the VNNI format
                    } else if (loop_port.get_dim_idx() == 0) {
                        auto k_blk_size = static_cast<int64_t>(brgemm_utils::compute_vnni_factor(precision));
                        loop_desc.ptr_increment =
                            snippets::utils::dynamic_safe_mul(loop_desc.ptr_increment, k_blk_size);
                        loop_desc.finalization_offset =
                            snippets::utils::dynamic_safe_mul(loop_desc.finalization_offset, k_blk_size);
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

    auto get_repacking_loop_idces = [](const snippets::lowered::ExpressionPtr& brgemm_expr) {
        // Repacking may be extracted outside the snippets kernel. In this case, brgemm parent expression is a
        // parameter.
        const auto& brgemm_in1 = brgemm_expr->get_input_port_connector(1)->get_source();
        const auto& shape_infer_seq = ov::snippets::utils::get_first_parent_shape_infer_expr_seq(brgemm_in1.get_expr());
        const auto source =
            shape_infer_seq.empty() ? brgemm_in1 : shape_infer_seq.back()->get_input_port_connector(0)->get_source();
        if (is_type<ov::op::v0::Parameter>(source.get_expr()->get_node())) {
            return std::vector<size_t>{};
        }
        const auto repacking_expr = brgemm_utils::repacking::get_copy_b_expr(brgemm_expr);
        OPENVINO_ASSERT(repacking_expr, "BrgemmCopyB expression is not found");
        return repacking_expr->get_loop_ids();
    };

    for (const auto& expr : linear_ir) {
        const auto brgemm = ov::as_type_ptr<BrgemmCPU>(expr->get_node());
        if (!brgemm || !brgemm_utils::with_repacking(brgemm->get_type())) {
            continue;
        }
        const auto& brgemm_loop_ids = expr->get_loop_ids();
        const auto& repacking_loop_ids = get_repacking_loop_idces(expr);
        // Continue if there is no blocking loop
        if (brgemm_loop_ids.empty() && repacking_loop_ids.empty()) {
            continue;
        }

        OPENVINO_ASSERT(brgemm_loop_ids.size() > repacking_loop_ids.size(), "Invalid BrgemmCopyB loop configuration");
        const auto& loop_manager = linear_ir.get_loop_manager();
        for (auto i = repacking_loop_ids.size(); i < brgemm_loop_ids.size(); i++) {
            const auto& loop = loop_manager->get_loop_info(brgemm_loop_ids[i]);
            auto uni_loop = ov::as_type_ptr<snippets::lowered::UnifiedLoopInfo>(loop);
            if (!uni_loop) {
                uni_loop = ov::as_type_ptr<snippets::lowered::ExpandedLoopInfo>(loop)->get_unified_loop_info();
            }
            if (!m_affected_loops.count(uni_loop) && update_loop_info(uni_loop)) {
                m_affected_loops.insert(uni_loop);
                modified = true;
            }
        }
    }

    return modified;
}
}  // namespace ov::intel_cpu
