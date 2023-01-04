// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/lowered/buffer_propagate_offset_and_reset.hpp"
#include "snippets/itt.hpp"

namespace ngraph {
namespace snippets {
namespace pass {
namespace lowered {

void PropagateOffsetAndResetBuffer::propagate_offset(const LoweredExprIR& linear_ir, const LoweredExprPtr& buffer_expr, const size_t offset) {
    // If Buffer has offset We set this offset in the next Load and Store ops
    // to correctly read and write data because all buffers have the one register
    // Also if user sets offset to a Buffer It means that the Buffer has the corresponding Load and Store ops

    const auto buffer = ov::as_type_ptr<op::Buffer>(buffer_expr->get_node());

    // Propagate to up: in Store. Buffer can have only one Store
    {
        if (buffer->is_intermediate_memory()) {
            OPENVINO_ASSERT(buffer_expr->get_inputs().size() == 1, "Buffer with intermediate memory must have one parent");
            auto parent_expr = linear_ir.get_expr_by_output(buffer_expr->get_inputs()[0]);
            auto parent_node = parent_expr->get_node();
            if (auto memory_access = ov::as_type_ptr<ngraph::snippets::op::MemoryAccess>(parent_node)) {
                memory_access->set_output_offset(offset, 0); // TODO
            } else {
                throw ngraph_error(
                        "Buffer::set_offset() was called when Buffer didn't have the corresponding MemoryAccess op for offset propagation");
            }
        }
    }
    // Propagate to down: in Load. Buffer can have several Load and Loops after himself. We should go through all target inputs
    const auto& buffer_out = buffer_expr->get_outputs()[0];
    for (const auto& child_expr : linear_ir.get_exprs_by_input(buffer_out)) {
        const auto& child_node = child_expr->get_node();
        if (auto memory_access = ov::as_type_ptr<ngraph::snippets::op::MemoryAccess>(child_node)) {
            memory_access->set_input_offset(offset, 0); // TODO
        } else {
            throw ngraph_error(
                    "Buffer::set_offset() was called when Buffer didn't have the corresponding MemoryAccess op for offset propagation");
        }
    }
}


bool PropagateOffsetAndResetBuffer::run(LoweredExprIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::buffer_propagate_offset_and_reset")
    std::vector<LoweredExprIR::container::iterator> exprs_to_del;
    bool modified = false;
    size_t offset = 0;
    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        if (auto buffer = as_type_ptr<op::Buffer>(expr_it->get()->get_node())) {
            const auto buffer_size = buffer->get_byte_size();
            // If it's the first buffer, offsets are zero => nothing to propagate, can continue
            if (m_buffer_scratchpad_size == 0) {
                m_buffer_scratchpad_size += buffer_size;
                continue;
            }

            if (buffer->is_intermediate_memory()) {
                const auto& parent_expr = linear_ir.get_expr_by_output(expr_it->get()->get_inputs()[0]);
                const auto& prent_node = parent_expr->get_node();
                // Brgemm is a special case, since it doesn't allow memory reuse
                if (ov::is_type<op::Brgemm>(prent_node)) {
                    offset = m_buffer_scratchpad_size;
                    buffer->set_offset(static_cast<int64_t>(offset));
                    propagate_offset(linear_ir, *expr_it, offset);
                    m_buffer_scratchpad_size += buffer_size;
                    continue;
                }
                const auto current_allocated_memory_size = m_buffer_scratchpad_size - offset;
                if (buffer_size > current_allocated_memory_size) {
                    m_buffer_scratchpad_size += (buffer_size - current_allocated_memory_size);
                    // Note: we don't update offset because we just add memory to needed size
                }
                propagate_offset(linear_ir, *expr_it, offset);
            } else {
                // Single Buffer without input should allocate new memory
                offset = m_buffer_scratchpad_size;
                buffer->set_offset(static_cast<int64_t>(offset));
                propagate_offset(linear_ir, *expr_it, offset);
                m_buffer_scratchpad_size += buffer_size;
            }
            modified = true;
        } else if (auto loop_end = as_type_ptr<op::LoopEnd>(expr_it->get()->get_node())) {
            // Note: Buffer always employ inplace logics by default. It means that if a loop has both
            // an input and an output connected to Buffers, the corresponding register should nevertheless be
            // incremented only once (because when the input reg is incremented, output incremented automatically).
            // This condition should be removed when Buffers stop being inplace by default.
            const auto& ins = expr_it->get()->get_inputs();
            std::vector<int> buffer_idx{};
            for (int i = 0; i < static_cast<int>(ins.size()) - 1; i++) {
                const auto& in = ins[i];
                // If producer of the input expr is buffer: this covers Buffer->Load patterns
                if (ov::is_type<op::Buffer>(linear_ir.get_expr_by_output(in)->get_node()))
                    buffer_idx.push_back(i);
                // If consumer of the input is buffer: Store->Buffer patterns
                for (const auto& consumer : linear_ir.get_exprs_by_input(in)) {
                    if (ov::is_type<op::Buffer>(consumer->get_node()))
                        buffer_idx.push_back(i);
                }
            }
            // This is currently not allowed because all Buffers are implicitly used in-place
            if (buffer_idx.size() > 2) {
                throw ngraph_error("More than 2 Buffers connected to a single LoopEnd.");
            } else if (buffer_idx.size() == 2) {
                const auto idx_to_drop = buffer_idx.front();
                auto ptr_increments = loop_end->get_ptr_increments();
                auto fin_offsets = loop_end->get_finalization_offsets();
                ptr_increments[idx_to_drop] = 0;
                fin_offsets[idx_to_drop] = 0;
                loop_end->set_ptr_increments(ptr_increments);
                loop_end->set_finalization_offsets(fin_offsets);
            }
        }
    }
    return modified;
}

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph
