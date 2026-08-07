// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "adjust_gemm_copy_b_loop_ports.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_info.hpp"
#include "snippets/lowered/loop_port.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/aarch64/op/gemm_cpu.hpp"
#include "transformations/snippets/aarch64/op/gemm_utils.hpp"
#include "transformations/snippets/common/pass/lowered/adjust_copy_b_loop_ports.hpp"

namespace ov::intel_cpu {

namespace {
int64_t get_rhs_packed_ptr_increment(const ov::element::Type& precision, size_t n_increment, size_t K) {
    if (snippets::utils::is_dynamic_value(n_increment) || snippets::utils::is_dynamic_value(K)) {
        return snippets::utils::get_dynamic_value<int64_t>();
    }

    // KAI packs RHS by N blocks: one bias value plus K RHS values per N lane.
    // Derive the snippets ptr_increment from KAI byte offsets; for current f32/f16 packers this reduces to K + 1.
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
                        pass::copy_b_loop_ports::assign_new_ptr_increment(new_ptr_increment, loop_desc);
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

    auto get_repacking_loop_idces = [](const snippets::lowered::ExpressionPtr& gemm_expr) {
        return pass::copy_b_loop_ports::get_repacking_loop_idces(
            gemm_expr,
            ov::intel_cpu::aarch64::gemm_utils::repacking::get_copy_b_expr,
            "GemmCopyB expression is not found");
    };

    auto is_target_expr = [](const snippets::lowered::ExpressionPtr& expr) {
        const auto gemm = ov::as_type_ptr<ov::intel_cpu::aarch64::GemmCPU>(expr->get_node());
        return static_cast<bool>(gemm);
    };

    modified = pass::copy_b_loop_ports::run(linear_ir,
                                            m_affected_loops,
                                            is_target_expr,
                                            get_repacking_loop_idces,
                                            pass::aarch64::AdjustGemmCopyBLoopPorts::update_loop_info,
                                            "Invalid GemmCopyB loop configuration");

    return modified;
}
}  // namespace ov::intel_cpu
