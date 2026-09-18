// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "adjust_copy_b_loop_ports.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_set>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/parameter.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_info.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/utils/utils.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu::pass {

void AdjustCopyBLoopPorts::assign_new_ptr_increment(int64_t new_ptr_increment,
                                                    snippets::lowered::UnifiedLoopInfo::LoopPortDesc& loop_desc) {
    const auto old_ptr_incr = loop_desc.ptr_increment;
    const auto old_final_offset = loop_desc.finalization_offset;

    if (none_of(old_ptr_incr, 0, new_ptr_increment)) {
        loop_desc.ptr_increment = new_ptr_increment;
        if (!ov::snippets::utils::is_dynamic_value(old_final_offset)) {
            OPENVINO_ASSERT(old_final_offset % old_ptr_incr == 0, "Can't rescale finalization offsets");
            loop_desc.finalization_offset =
                ov::snippets::utils::dynamic_safe_mul(loop_desc.ptr_increment, (old_final_offset / old_ptr_incr));
        }
    }
}

bool AdjustCopyBLoopPorts::run(const snippets::lowered::LinearIR& linear_ir) {
    bool modified = false;
    for (const auto& expr : linear_ir) {
        if (!is_target_expr(expr)) {
            continue;
        }

        const auto& gemm_loop_ids = expr->get_loop_ids();
        const auto& gemm_in1 = expr->get_input_port_connector(1)->get_source();
        const auto& shape_infer_seq = ov::snippets::utils::get_first_parent_shape_infer_expr_seq(gemm_in1.get_expr());
        const auto source =
            shape_infer_seq.empty() ? gemm_in1 : shape_infer_seq.back()->get_input_port_connector(0)->get_source();
        std::vector<size_t> repacking_loop_ids;
        if (!is_type<ov::op::v0::Parameter>(source.get_expr()->get_node())) {
            const auto repacking_expr = get_copy_b_expr(expr);
            OPENVINO_ASSERT(repacking_expr, copy_b_not_found_message());
            repacking_loop_ids = repacking_expr->get_loop_ids();
        }
        if (gemm_loop_ids.empty() && repacking_loop_ids.empty()) {
            continue;
        }

        OPENVINO_ASSERT(gemm_loop_ids.size() > repacking_loop_ids.size(), invalid_loop_config_message());
        const snippets::lowered::LoopManagerPtr& loop_manager = linear_ir.get_loop_manager();
        for (auto i = repacking_loop_ids.size(); i < gemm_loop_ids.size(); i++) {
            const auto& loop = loop_manager->get_loop_info(gemm_loop_ids[i]);
            auto uni_loop = ov::as_type_ptr<snippets::lowered::UnifiedLoopInfo>(loop);
            if (!uni_loop) {
                uni_loop = ov::as_type_ptr<snippets::lowered::ExpandedLoopInfo>(loop)->get_unified_loop_info();
            }
            if (!m_affected_loops.count(uni_loop) && update_loop_info_impl(uni_loop)) {
                m_affected_loops.insert(uni_loop);
                modified = true;
            }
        }
    }
    return modified;
}

}  // namespace ov::intel_cpu::pass
