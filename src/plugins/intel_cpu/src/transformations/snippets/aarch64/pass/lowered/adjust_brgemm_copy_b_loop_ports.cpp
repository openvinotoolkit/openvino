// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "adjust_brgemm_copy_b_loop_ports.hpp"

#include "snippets/itt.hpp"
#include "snippets/lowered/expressions/buffer_expression.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/aarch64/op/gemm_copy_b.hpp"
#include "transformations/snippets/aarch64/op/gemm_cpu.hpp"

namespace ov::intel_cpu {

bool pass::aarch64::AdjustBrgemmCopyBLoopPorts::update_loop_info(
    const std::shared_ptr<snippets::lowered::UnifiedLoopInfo>& loop_info) {
    OPENVINO_ASSERT(loop_info, "Invalid loop info pointer");
    bool modified = false;
    auto caller = [&](snippets::lowered::LoopPort& loop_port,
                      snippets::lowered::UnifiedLoopInfo::LoopPortDesc& loop_desc) {
        const auto& p = *loop_port.get_expr_port();
        if (p.get_type() == snippets::lowered::ExpressionPort::Input && p.get_index() == 1) {
            const auto& node = p.get_expr()->get_node();
            if (auto brg = as_type_ptr<ov::intel_cpu::aarch64::GemmCPU>(node)) {
                // from format KN to NK64n(64 is n block), and for each K64n, repack to nK8n
                std::cout << "loop_port.get_dim_idx():" << loop_port.get_dim_idx() << std::endl;
                if (loop_port.is_incremented()) {
                    // N blocking loop
                    if (loop_port.get_dim_idx() == 0) {
                        std::cout << "loop_port.get_dim_idx() == 0" << std::endl;
                        // int64_t k_blk_size = 16; // K dimension if k w/o block
                        const auto& in_0_shape = brg->get_input_shape(0);
                        int64_t k_blk_size = in_0_shape.back(); // K dimension if k w/o block
                        std::cout << "K:" << k_blk_size << std::endl;
                        // ptr_increment is 1, inc is 64.
                        loop_desc.ptr_increment =
                            snippets::utils::dynamic_safe_mul(loop_desc.ptr_increment, (k_blk_size + 1));
                        loop_desc.finalization_offset =
                            snippets::utils::dynamic_safe_mul(loop_desc.finalization_offset, (k_blk_size + 1));
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

bool pass::aarch64::AdjustBrgemmCopyBLoopPorts::run(const snippets::lowered::LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::AdjustBrgemmCopyBLoopPorts")

    bool modified = false;

    // auto get_repacking_loop_idces = [](const snippets::lowered::ExpressionPtr& brgemm_expr) {
    //     // Repacking may be extracted outside the snippets kernel. In this case, brgemm parent expression is a
    //     // parameter.
    //     const auto& brgemm_in1 = brgemm_expr->get_input_port_connector(1)->get_source();
    //     const auto& shape_infer_seq = ov::snippets::utils::get_first_parent_shape_infer_expr_seq(brgemm_in1.get_expr());
    //     const auto source =
    //         shape_infer_seq.empty() ? brgemm_in1 : shape_infer_seq.back()->get_input_port_connector(0)->get_source();
    //     if (is_type<ov::op::v0::Parameter>(source.get_expr()->get_node())) {
    //         return std::vector<size_t>{};
    //     }
    //     const auto repacking_expr = brgemm_utils::repacking::get_copy_b_expr(brgemm_expr);
    //     OPENVINO_ASSERT(repacking_expr, "BrgemmCopyB expression is not found");
    //     return repacking_expr->get_loop_ids();
    // };

    for (const auto& expr : linear_ir) {
        const auto brgemm = ov::as_type_ptr<ov::intel_cpu::aarch64::GemmCPU>(expr->get_node());
        if (!brgemm) {
            continue;
        }
        std::cout << "brgemm1:" << std::endl;
        const auto& brgemm_loop_ids = expr->get_loop_ids();
        std::cout << "brgemm_loop_ids.size:" << brgemm_loop_ids.size() << std::endl;
        // const auto& repacking_loop_ids = get_repacking_loop_idces(expr);
        // // Continue if there is no blocking loop
        // if (brgemm_loop_ids.empty() && repacking_loop_ids.empty()) {
        //     continue;
        // }

        // OPENVINO_ASSERT(brgemm_loop_ids.size() > repacking_loop_ids.size(), "Invalid BrgemmCopyB loop configuration");
        const auto& loop_manager = linear_ir.get_loop_manager();
        // for (auto i = repacking_loop_ids.size(); i < brgemm_loop_ids.size(); i++) {
        for (auto i = 1; i < brgemm_loop_ids.size(); i++) {
            std::cout << "brgemm_loop_ids:" << i << std::endl;
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
