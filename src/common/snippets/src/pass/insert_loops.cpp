// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>
#include "snippets/pass/insert_loops.hpp"
#include "snippets/pass/loop_helpers.hpp"

#include <ngraph/rt_info.hpp>

ngraph::snippets::pass::InsertLoops::InsertLoops(ov::PartialShape master_shape, size_t vector_size)
: master_shape(std::move(master_shape)), vector_size(vector_size) {
}

bool ngraph::snippets::pass::InsertLoops::run_on_model(const std::shared_ptr<ov::Model> &model) {
    RUN_ON_FUNCTION_SCOPE(InsertLoops);
    if (master_shape.is_dynamic())
        throw ngraph_error("InsertLoops doesn't support dynamic shapes yet");

    const auto inner_dim = master_shape.size() - 1;
    // Note: outer_dim could overflow if master_shape.size() < 2
    const auto outer_dim = master_shape.size() - 2;
    const auto inner_work_amount = master_shape[inner_dim].get_length();
    const auto outer_work_amount = master_shape.size() >= 2 ? master_shape[outer_dim].get_length() : 1;

    ParameterVector commonParams = model->get_parameters();
    // Note that topological sort parses node arguments in reversed order, but results are added  - in direct order
    // So ve need to pass the reversed results to LoopEnd to keep the original traversal order in topological sorter
    const auto& orig_results = model->get_results();
    ResultVector commonResults(orig_results.rbegin(), orig_results.rend());
    std::vector<PartialShape> ioShapes;
    ioShapes.reserve(commonParams.size() + commonResults.size());
    std::transform(commonParams.begin(), commonParams.end(), std::back_inserter(ioShapes),
                   [](const std::shared_ptr<Node>& n) { return n->get_output_partial_shape(0); });
    std::transform(commonResults.begin(), commonResults.end(), std::back_inserter(ioShapes),
                   [](const std::shared_ptr<Node>& n) { return n->get_input_partial_shape(0); });

    if (inner_work_amount > 0) {
        std::vector<bool> apply_increments;
        apply_increments.reserve(ioShapes.size());
        // Inner Loop applies increments if a dimension is not broadcasted
        std::transform(ioShapes.begin(), ioShapes.end(), std::back_inserter(apply_increments),
                       [=](const PartialShape& ps) {
                           return ps[inner_dim] != 1 && master_shape[inner_dim] != 1;
                       });
        std::vector<int64_t> inner_finalization_offsets(ioShapes.size(), 0);
        if (outer_work_amount > 1) {
            // We need to step back if an outer dim is broadcasted, while the corresponding lower one is not
            std::transform(ioShapes.begin(), ioShapes.end(), inner_finalization_offsets.begin(),
                           [=](const PartialShape& ps) {
                               return ps[outer_dim] == 1 && ps[inner_dim] != 1 ? -inner_work_amount : 0;
                           });
        }
        const auto& inner_loop_begin = op::insertLoopBegin(commonParams);
        const auto& inner_loop_end = insertLoopEnd(commonResults, inner_loop_begin, inner_work_amount,
                                                   vector_size, apply_increments,  inner_finalization_offsets);
        // set internal flag to enable scalar vs vector loop optimizations
        inner_loop_end->has_outer_loop = outer_work_amount > 1;
        // Due to features of topological sort, some Constants (Scalars) may appear right after Parameters in
        // sorted ops (so it's between Parameters and LoopBegin). Consequently, ScalarEmitters would be called
        // outside the Loop, and only the first Loop iteration would yield correct data (assuming the vector reg
        // assigned to scalar will get corrupted inside the loop body). To avoid such cases, we add control dependency
        // on LoopBegin to guarantee that the constants are executed inside the Loop.
        for (const auto& n : model->get_ordered_ops()) {
            if (auto c = std::dynamic_pointer_cast<ov::op::v0::Constant>(n))
                c->add_control_dependency(inner_loop_begin);
            else if (n == inner_loop_begin)
                break;
        }
    }

    if (outer_work_amount > 1) {
        std::vector<bool> apply_increments;
        apply_increments.reserve(ioShapes.size());
        // Outer Loop applies increments only if a corresponding lower dim was broadcasted (or all lower dims == 1)
        std::transform(ioShapes.begin(), ioShapes.end(), std::back_inserter(apply_increments),
                       [=](const PartialShape& ps) {
                           return ps[outer_dim] != 1 && ps[inner_dim] == 1;
                       });
        const auto& outer_loop_begin = op::insertLoopBegin(commonParams);
        insertLoopEnd(commonResults, outer_loop_begin, outer_work_amount, 1, apply_increments);
    }

    return true;
}