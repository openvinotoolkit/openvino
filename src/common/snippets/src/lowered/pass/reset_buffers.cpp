// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/reset_buffers.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

namespace ngraph {
namespace snippets {
namespace lowered {
namespace pass {

bool ResetBuffers::reuse_buffer_increments(const LinearIR& linear_ir, const ExpressionPtr& loop_end_expr) {
    const auto loop_end = ov::as_type_ptr<op::LoopEnd>(loop_end_expr->get_node());
    if (!loop_end)
        return false;

    const auto loop_tds = loop_end_expr->get_input_tensors();
    const auto input_count = loop_end->get_input_num();
    const auto output_count = loop_end->get_output_num();

    std::set<size_t> resetting_buffers;
    std::set<size_t> buffers_ids;
    for (size_t i = 0; i < input_count; ++i) {
        const auto& parent_output = loop_tds[i]->get_source().get_expr();
        if (const auto buffer = ov::as_type_ptr<op::Buffer>(parent_output->get_node())) {
            // If Buffer is missed in set, Just save - it's first meeting
            if (buffers_ids.count(buffer->get_id()) == 0) {
                buffers_ids.insert(buffer->get_id());
            } else {
                // The Buffer with the same ID is in set - need to add this Buffer idx to set of Buffers for resetting
                resetting_buffers.insert(i);
            }
        }
    }
    for (size_t i = 0; i < output_count; ++i) {
        const auto consumer_inputs = loop_tds[input_count + i]->get_consumers();
        size_t buffer_count = 0;
        size_t loop_count = 0;
        for (const auto& consumer_input : consumer_inputs) {
            const auto& child_node = consumer_input.get_expr()->get_node();
            if (const auto buffer = ov::as_type_ptr<op::Buffer>(child_node)) {
                buffer_count++;
                // If Buffer is missed in set, Just save - it's first meeting
                if (buffers_ids.count(buffer->get_id()) == 0) {
                    buffers_ids.insert(buffer->get_id());
                } else {
                    // The Buffer with the same ID is in set - need to add this Buffer idx to set of Buffers for resetting
                    resetting_buffers.insert(input_count + i);
                }
            } else if (ov::is_type<op::LoopEnd>(child_node)) {
                loop_count++;
            }
        }
        if (buffer_count > 0) {
            OPENVINO_ASSERT((buffer_count == 1) && (buffer_count + loop_count == consumer_inputs.size()),
                            "Loop output must have not more than 1 Buffer");
        }
    }

    if (resetting_buffers.empty())
        return false;

    auto new_ptr_increments = loop_end->get_ptr_increments();
    auto new_finalization_offsets = loop_end->get_finalization_offsets();
    for (auto idx_to_drop : resetting_buffers) {
        new_ptr_increments[idx_to_drop] = 0;
        new_finalization_offsets[idx_to_drop] = 0;
    }
    loop_end->set_ptr_increments(new_ptr_increments);
    loop_end->set_finalization_offsets(new_finalization_offsets);
    return true;
}

bool ResetBuffers::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::ResetBuffers")
    bool modified = false;

    for (const auto& expr : linear_ir) {
        const auto& node = expr->get_node();
        if (ov::is_type<op::LoopEnd>(node)) {
            modified |= reuse_buffer_increments(linear_ir, expr);
        }
    }

    return modified;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ngraph
