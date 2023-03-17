// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/lowered/cleanup_loop_offsets.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

namespace ngraph {
namespace snippets {
namespace pass {
namespace lowered {

bool CleanupLoopOffsets::run(LoweredExprIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::CleanupLoopOffsets")
    if (linear_ir.empty())
        return false;
    bool is_modified = false;
    // Note: it doesn't make sense to check the last expression - it must always be Result
    const auto before_last = std::prev(linear_ir.end());
    for (auto expr_it = linear_ir.begin(); expr_it != before_last; expr_it++) {
        const auto& node = expr_it->get()->get_node();
        if (auto loop_end = as_type_ptr<op::LoopEnd>(node)) {
                auto next_expr_it = std::next(expr_it);
                const auto& next_node = next_expr_it->get()->get_node();
                // Note: Finalization offsets before the Result can be safely disregarded
                // TODO: Need verify that Buffers on the inputs doesn't have other consumers (other Loops)
                //       and this Loop doesn't have Buffer on other outputs.
                if (is_type<ngraph::op::v0::Result>(next_node)) {
                    const auto& fin_offsets = loop_end->get_finalization_offsets();
                    loop_end->set_finalization_offsets(std::vector<int64_t>(fin_offsets.size(), 0));
                    is_modified = true;
                }
                if (auto outer_loop_end = as_type_ptr<op::LoopEnd>(next_node)) {
                    auto fin_offsets = loop_end->get_finalization_offsets();
                    std::unordered_map<TensorDescriptorPtr, size_t> per_tensor_offset;
                    const auto& loop_inputs = expr_it->get()->get_inputs();
                    for (size_t i = 0; i < fin_offsets.size(); i++)
                        per_tensor_offset[loop_inputs[i]] = i;

                    auto outer_ptr_increments = outer_loop_end->get_ptr_increments();
                    const auto& outer_loop_inputs = next_expr_it->get()->get_inputs();
                    for (size_t i = 0; i < outer_ptr_increments.size(); i++) {
                        const auto& managed_tensor = outer_loop_inputs[i];
                        const auto& found = per_tensor_offset.find(managed_tensor);
                        if (found != per_tensor_offset.end()) {
                            outer_ptr_increments[i] += fin_offsets[found->second];
                            fin_offsets[found->second] = 0;
                            is_modified = true;
                        }
                    }
                    outer_loop_end->set_ptr_increments(outer_ptr_increments);
                    loop_end->set_finalization_offsets(fin_offsets);
                }
        }
    }
    return is_modified;
}

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph

