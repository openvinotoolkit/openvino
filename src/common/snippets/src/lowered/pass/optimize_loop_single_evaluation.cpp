// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/optimize_loop_single_evaluation.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

bool OptimizeLoopSingleEvaluation::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::OptimizeLoopSingleEvaluation")
    if (linear_ir.empty())
        return false;

    bool is_modified = false;
    for (const auto& expr : linear_ir) {
        if (auto loop_end = ov::as_type_ptr<op::LoopEndStatic>(expr->get_node())) {
            // *1* solo vector/tail loop + empty outer loop
            //      => skip increments (both counter & ptr) : set evaluate_once flag
            // *2* solo vector/tail loop + non-empty outer loop
            //      => skip counter increments but perform ptr increments : set evaluate_once,
            //         and perform pointer increments through finalization offsets
            // *3* vector loop(s) + one tail loop
            //      => vector as usual, tail depends on outer loop, see *1* and *2*
            if (loop_end->get_work_amount() >= 2 * loop_end->get_increment())
                continue;

            auto new_finalization_offsets = loop_end->get_finalization_offsets();
            const auto& ptr_increments = loop_end->get_ptr_increments();
            const auto work_amount_incr = static_cast<int64_t>(loop_end->get_increment());
            for (size_t i = 0; i < new_finalization_offsets.size(); i++) {
                new_finalization_offsets[i] += ptr_increments[i] * work_amount_incr;
            }
            loop_end->set_finalization_offsets(new_finalization_offsets);
            loop_end->set_ptr_increments(std::vector<int64_t>(new_finalization_offsets.size(), 0));
            loop_end->set_evaluate_once(true);
            is_modified = true;
        }
    }
    return is_modified;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov

