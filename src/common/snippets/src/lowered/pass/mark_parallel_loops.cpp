// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/mark_parallel_loops.hpp"

#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils/utils.hpp"

namespace ov::snippets::lowered::pass {
bool MarkParallelLoops::run(LinearIR& linear_ir,
                            lowered::LinearIR::constExprIt begin,
                            lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::MarkParallelLoops")
    size_t outermost_loop_id = SIZE_MAX;
    // todo: currently, we use the simplest possible strategy: convert only the outermost loop to parallel.
    //  a more sophisticaled scheduling might be required in some cases, e.g. parallel loops over N dimension in MLP
    for (auto expr_it = begin; expr_it != end; expr_it++) {
        const auto expr = *expr_it;
        const auto& expr_loops = expr->get_loop_ids();
        if (!expr_loops.empty()) {
            if (outermost_loop_id == SIZE_MAX)
                outermost_loop_id = expr_loops.front();
            else
                OPENVINO_ASSERT(outermost_loop_id == expr_loops.front(),
                                "This LIR is not supported for scheduling yet");
        }
    }
    OPENVINO_ASSERT(outermost_loop_id != SIZE_MAX, "Failed to find outermost loop in LIR");
    const auto& loop_manager = linear_ir.get_loop_manager();
    loop_manager->get_loop_info(outermost_loop_id)->set_is_parallel(true);
    return true;
}

}  // namespace ov::snippets::lowered::pass
