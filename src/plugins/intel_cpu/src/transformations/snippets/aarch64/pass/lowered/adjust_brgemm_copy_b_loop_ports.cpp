// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "adjust_brgemm_copy_b_loop_ports.hpp"

#include "snippets/itt.hpp"
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
                if (loop_port.is_incremented()) {
                    if (loop_port.get_dim_idx() == 0) {
                        // N blocking loop
                        const auto& in_0_shape = brg->get_input_shape(0);
                        int64_t K = in_0_shape.back();  // K dimension(K is not blocked)
                        // NK repacked and padded to to N(K+1)
                        // ptr_increment is 1, inc is 64. inc=inc*ptr_increment=64*(K+1)
                        loop_desc.ptr_increment = snippets::utils::dynamic_safe_mul(loop_desc.ptr_increment, (K + 1));
                        loop_desc.finalization_offset =
                            snippets::utils::dynamic_safe_mul(loop_desc.finalization_offset, (K + 1));
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

    for (const auto& expr : linear_ir) {
        const auto brgemm = ov::as_type_ptr<ov::intel_cpu::aarch64::GemmCPU>(expr->get_node());
        if (!brgemm) {
            continue;
        }
        const auto& brgemm_loop_ids = expr->get_loop_ids();
        if (brgemm_loop_ids.empty()) {
            continue;
        }
        const auto& loop_manager = linear_ir.get_loop_manager();
        // only adjust inner most loop(N loop)
        const auto& loop = loop_manager->get_loop_info(brgemm_loop_ids.back());
        auto uni_loop = ov::as_type_ptr<snippets::lowered::UnifiedLoopInfo>(loop);
        if (!uni_loop) {
            uni_loop = ov::as_type_ptr<snippets::lowered::ExpandedLoopInfo>(loop)->get_unified_loop_info();
        }
        if (!m_affected_loops.count(uni_loop) && update_loop_info(uni_loop)) {
            m_affected_loops.insert(uni_loop);
            modified = true;
        }
    }

    return modified;
}
}  // namespace ov::intel_cpu
