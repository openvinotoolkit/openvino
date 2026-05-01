// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "adjust_gemm_copy_b_loop_ports.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_info.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/lowered/loop_port.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/aarch64/op/gemm_cpu.hpp"
#include "transformations/snippets/aarch64/op/gemm_utils.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu {

namespace {
void assign_new_ptr_increment(int64_t new_ptr_increment,
                              ov::snippets::lowered::UnifiedLoopInfo::LoopPortDesc& loop_desc) {
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

int64_t get_rhs_packed_ptr_increment(const ov::element::Type& precision, size_t n_increment, size_t K) {
    if (snippets::utils::is_dynamic_value(n_increment) || snippets::utils::is_dynamic_value(K)) {
        return snippets::utils::get_dynamic_value<int64_t>();
    }

    const auto n_step = aarch64::gemm_utils::repacking::get_rhs_packed_n_step(precision);
    OPENVINO_ASSERT(n_increment % n_step == 0, "GEMM N loop increment must be aligned with KAI RHS packed N step");

    const auto element_size = precision.size();
    const auto packed_offset = aarch64::gemm_utils::repacking::get_rhs_packed_offset(precision, n_increment, K);
    const auto loop_increment_size = n_increment * element_size;
    OPENVINO_ASSERT(loop_increment_size != 0 && packed_offset % loop_increment_size == 0,
                    "KAI RHS packed offset can't be represented as a loop pointer increment");

    const auto ptr_increment = packed_offset / loop_increment_size;
    OPENVINO_ASSERT(ptr_increment <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
                    "KAI RHS packed pointer increment is out of int64_t range");
    return static_cast<int64_t>(ptr_increment);
}
}  // namespace

bool pass::aarch64::AdjustGemmCopyBLoopPorts::update_loop_info(
    const std::shared_ptr<snippets::lowered::UnifiedLoopInfo>& loop_info) {
    OPENVINO_ASSERT(loop_info, "Invalid loop info pointer");
    bool modified = false;
    auto caller = [&](snippets::lowered::LoopPort& loop_port,
                      snippets::lowered::UnifiedLoopInfo::LoopPortDesc& loop_desc) {
        const auto& p = *loop_port.get_expr_port();
        if (p.get_type() == snippets::lowered::ExpressionPort::Input && p.get_index() == 1) {
            const auto& expr = p.get_expr();
            if (as_type_ptr<ov::intel_cpu::aarch64::GemmCPU>(expr->get_node())) {
                // GemmCopyB packs RHS outside the N blocking loop, so the GemmCPU B port must step through the
                // KAI packed RHS layout rather than the original KN tensor.
                if (loop_port.is_incremented()) {
                    if (loop_port.get_dim_idx() == 0) {
                        const auto& b_shape = ov::snippets::utils::get_planar_vdims(*loop_port.get_expr_port());
                        OPENVINO_ASSERT(b_shape.size() >= 2, "GemmCPU B input must have at least 2 dimensions");
                        const auto K = *++b_shape.rbegin();
                        const auto& precision = expr->get_node()->get_input_element_type(1);
                        const auto new_ptr_increment =
                            get_rhs_packed_ptr_increment(precision, loop_info->get_increment(), K);
                        assign_new_ptr_increment(new_ptr_increment, loop_desc);
                    } else {
                        OPENVINO_THROW("Unexpected loop port dimension index in AdjustGemmCopyBLoopPorts");
                    }
                    modified = true;
                }
            }
        }
    };
    loop_info->iterate_through_infos(caller);
    return modified;
}

bool pass::aarch64::AdjustGemmCopyBLoopPorts::run(const snippets::lowered::LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::AdjustGemmCopyBLoopPorts")

    bool modified = false;

    for (const auto& expr : linear_ir) {
        const auto gemm = ov::as_type_ptr<ov::intel_cpu::aarch64::GemmCPU>(expr->get_node());
        if (!gemm) {
            continue;
        }
        const auto& gemm_loop_ids = expr->get_loop_ids();
        if (gemm_loop_ids.empty()) {
            continue;
        }
        const auto& loop_manager = linear_ir.get_loop_manager();
        // only adjust inner most loop(N loop)
        const auto& loop = loop_manager->get_loop_info(gemm_loop_ids.back());
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
