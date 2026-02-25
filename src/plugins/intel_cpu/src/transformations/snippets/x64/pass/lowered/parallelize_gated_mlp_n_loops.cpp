// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "parallelize_gated_mlp_n_loops.hpp"

#include <cstddef>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "openvino/itt.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"

using namespace ov::snippets::lowered;

namespace ov::intel_cpu::pass {

bool ParallelizeGatedMlpNLoops::run(LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::ParallelizeGatedMlpNLoops")

    std::vector<ExpressionPtr> brgemm_expressions;
    for (auto it = begin; it != end; ++it) {
        const auto& expr = *it;
        if (ov::is_type<ov::intel_cpu::BrgemmCPU>(expr->get_node())) {
            brgemm_expressions.push_back(expr);
        }
    }

    // Gated MLP pattern contains 3 brgemms, 2 first brgemms have the same A input
    const bool is_gated_mlp = brgemm_expressions.size() == 3 && brgemm_expressions[0]->get_input_expr_ptr(0) ==
                                                                    brgemm_expressions[1]->get_input_expr_ptr(0);
    if (!is_gated_mlp) {
        return false;
    }

    bool status = false;
    const auto& loop_manager = linear_ir.get_loop_manager();
    for (const auto& brgemm : brgemm_expressions) {
        const auto& out_subtensor = brgemm->get_output_port_descriptor(0)->get_subtensor();
        OPENVINO_ASSERT(out_subtensor.size() == 2, "Brgemm out subtensor should be 2D");
        // If there are no blocking loop by N, then we have nothing to parallelize
        if (ov::snippets::utils::is_full_dim_value(out_subtensor.back())) {
            continue;
        }

        // Bloking loop order: M -> N -> K
        // so N loop can be first(if M loop is absent) or second.
        const size_t n_loop_idx = ov::snippets::utils::is_full_dim_value(*(out_subtensor.rbegin() + 1)) ? 0 : 1;
        const auto& loop_idces = brgemm->get_loop_ids();
        OPENVINO_ASSERT(loop_idces.size() > n_loop_idx, "BrgemmCPU expr have N blocking loop");
        const auto n_loop_info = loop_manager->get_loop_info(loop_idces[n_loop_idx]);
        n_loop_info->set_is_parallel(true);
        status = true;
    }
    return status;
}

}  // namespace ov::intel_cpu::pass
