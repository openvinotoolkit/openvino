// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/cleanup_loop_offsets.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

bool CleanupLoopOffsets::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::CleanupLoopOffsets")
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
                if (is_type<ov::op::v0::Result>(next_node)) {
                    const auto& fin_offsets = loop_end->get_finalization_offsets();
                    loop_end->set_finalization_offsets(std::vector<int64_t>(fin_offsets.size(), 0));
                    is_modified = true;
                }
                if (auto outer_loop_end = as_type_ptr<op::LoopEnd>(next_node)) {
                    auto fin_offsets = loop_end->get_finalization_offsets();
                    std::unordered_map<PortConnectorPtr, size_t> per_port_connector_offset;
                    const auto& loop_inputs = expr_it->get()->get_input_port_connectors();
                    for (size_t i = 0; i < fin_offsets.size(); i++)
                        per_port_connector_offset[loop_inputs[i]] = i;

                    const auto outer_increment = static_cast<int64_t>(outer_loop_end->get_increment());
                    auto outer_ptr_increments = outer_loop_end->get_ptr_increments();
                    const auto& outer_loop_inputs = next_expr_it->get()->get_input_port_connectors();
                    for (size_t i = 0; i < outer_ptr_increments.size(); i++) {
                        const auto& managed_connector = outer_loop_inputs[i];
                        const auto& found = per_port_connector_offset.find(managed_connector);
                        if (found != per_port_connector_offset.end()) {
                            // Since data ptr is incremented on [ptr_increment x increment],
                            // we should guarantee proportionality of ptr shifts
                            // For example,
                            // Inner Loop: WA = 32, Inc = 1, ptr_increment[0] = 20, final_offset[0] = -640
                            // Outer Loop: WA = 70, Inc = 32, ptr_increment[0] = 20, final_offset[0] = -1400
                            // To save data ptr shift proportionality, we have to calculate so:
                            //    outer_ptr_increment[0] = (inner_final_offset[0] + outer_ptr_increment[0] * outer_Inc) / outer_Inc
                            //    outer_ptr_increment[0] = (-640 + 20 x 32) / 32 = 0
                            outer_ptr_increments[i] = (fin_offsets[found->second] + outer_ptr_increments[i] * outer_increment) / outer_increment;
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

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov

