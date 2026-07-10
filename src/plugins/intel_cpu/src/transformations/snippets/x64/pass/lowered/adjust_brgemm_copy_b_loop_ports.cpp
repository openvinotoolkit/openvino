// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "adjust_brgemm_copy_b_loop_ports.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "openvino/itt.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_info.hpp"
#include "snippets/lowered/loop_port.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/common/pass/lowered/adjust_copy_b_loop_ports.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/snippets/x64/op/brgemm_utils.hpp"

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
                const auto& brgemm_config = brg->get_config();
                /*
                 * The BrgemmCopyB operation repacks the weights in the following way:
                 *   1) Not-FP32 Brgemm requires inner K block which is equal to VNNI factor
                 *   2) If BrgemmConfig returns `are_wei_blocked()=1`, weights are repacked
                 *      in format BA<wei_k_blk>a<m_wei_n_blk>b<vnni_factor>a (if there is vnni).
                 *   3) If BrgemmConfig returns `are_wei_blocked()=0`, there is zero padding
                 *      for K and N dimensions due to memory access patern in BrgemmCopyB onednn kernel.
                 */
                if (brgemm_config.with_wei_repacking() && loop_port.is_incremented()) {
                    int64_t blocked_shape_ptr_inc = 0;
                    if (loop_port.get_dim_idx() == 1) {
                        // Blocking loop by K dimension:
                        blocked_shape_ptr_inc =
                            brgemm_utils::repacking::compute_K_blocked_stride(loop_desc.ptr_increment,
                                                                              brgemm_config.wei_n_blk(),
                                                                              brgemm_config.are_wei_blocked());
                    } else if (loop_port.get_dim_idx() == 0) {
                        // Blocking loop by N dimension:
                        // Attention: blocked_shape_ptr_inc is int64_t while K dimension is uint64_t
                        const auto& shape = loop_port.get_expr_port()->get_descriptor_ptr()->get_shape();
                        const auto K_dim = *++shape.rbegin();
                        if (snippets::utils::is_dynamic_value(K_dim)) {
                            blocked_shape_ptr_inc = snippets::utils::get_dynamic_value<int64_t>();
                        } else {
                            blocked_shape_ptr_inc =
                                brgemm_utils::repacking::compute_N_blocked_stride(static_cast<int64_t>(K_dim),
                                                                                  brgemm_config.wei_k_blk(),
                                                                                  brgemm_config.wei_dt(),
                                                                                  brgemm_config.are_wei_blocked());
                        }
                    } else {
                        OPENVINO_THROW("Unexpected loop port dimension index in AdjustBrgemmCopyBLoopPorts");
                    }

                    pass::copy_b_loop_ports::assign_new_ptr_increment(blocked_shape_ptr_inc, loop_desc);
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
        return pass::copy_b_loop_ports::get_repacking_loop_idces(brgemm_expr,
                                                                 brgemm_utils::repacking::get_copy_b_expr,
                                                                 "BrgemmCopyB expression is not found");
    };

    auto is_target_expr = [](const snippets::lowered::ExpressionPtr& expr) {
        const auto brgemm = ov::as_type_ptr<BrgemmCPU>(expr->get_node());
        return brgemm && brgemm->get_config().with_wei_repacking();
    };

    modified = pass::copy_b_loop_ports::run(linear_ir,
                                            m_affected_loops,
                                            is_target_expr,
                                            get_repacking_loop_idces,
                                            pass::AdjustBrgemmCopyBLoopPorts::update_loop_info,
                                            "Invalid BrgemmCopyB loop configuration");

    return modified;
}
}  // namespace ov::intel_cpu
