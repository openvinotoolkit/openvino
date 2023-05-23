// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/allocate_buffers.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

void AllocateBuffers::propagate_offset(const LinearIR& linear_ir, const ExpressionPtr& buffer_expr, const size_t offset) {
    // If Buffer has offset We set this offset in the connected MemoryAccess ops
    // to correctly read and write data because all Buffers has the common data pointer on buffer scratchpad

    const auto buffer = ov::as_type_ptr<op::Buffer>(buffer_expr->get_node());

    // Propagate to up: in Store. Buffer can have only one Store
    {
        if (buffer->is_intermediate_memory()) {
            OPENVINO_ASSERT(buffer_expr->get_input_port_connectors().size() == 1, "Buffer with intermediate memory must have one parent");
            const auto& parent_output = buffer_expr->get_input_port_connector(0)->get_source();
            const auto& parent_expr = parent_output.get_expr();
            const auto port = parent_output.get_index();
            const auto& parent_node = parent_expr->get_node();
            auto memory_access = ov::as_type_ptr<ov::snippets::op::MemoryAccess>(parent_node);
            if (memory_access && memory_access->is_memory_access_output_port(port)) {
                memory_access->set_output_offset(offset, port);
            } else {
                OPENVINO_THROW(
                        "Buffer::set_offset() was called when Buffer didn't have the corresponding MemoryAccess op for offset propagation");
            }
        }
    }
    // Propagate to down: in Load. Buffer can have several Load
    const auto& buffer_out = buffer_expr->get_output_port_connector(0);
    for (const auto& child_expr_input : buffer_out->get_consumers()) {
        const auto& child_expr = child_expr_input.get_expr();
        const auto port = child_expr_input.get_index();
        const auto& child_node = child_expr->get_node();
        auto memory_access = ov::as_type_ptr<ov::snippets::op::MemoryAccess>(child_node);
        if (memory_access && memory_access->is_memory_access_input_port(port)) {
            memory_access->set_input_offset(offset, port);
        } else if (ov::is_type<op::LoopEnd>(child_node)) {
            // After Loop initialization, Buffer can be connected to LoopEnd - it's ok
            continue;
        } else {
            OPENVINO_THROW(
                    "Buffer::set_offset() was called when Buffer didn't have the corresponding MemoryAccess op for offset propagation");
        }
    }
}


bool AllocateBuffers::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::AllocateBuffers");

    bool modified = false;
    size_t offset = 0;
    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        const auto& expr = *expr_it;
        if (auto buffer = as_type_ptr<op::Buffer>(expr->get_node())) {
            const auto buffer_size = buffer->get_byte_size();
            // If it's the first buffer, offsets are zero => nothing to propagate, can continue
            if (m_buffer_scratchpad_size == 0) {
                m_buffer_scratchpad_size += buffer_size;
                continue;
            }

            if (buffer->is_intermediate_memory()) {
                const auto& parent_expr = expr->get_input_port_connector(0)->get_source().get_expr();
                const auto& parent_node = parent_expr->get_node();
                // Full MemoryAccess ops need new memory. Previous logic is to check for parent isn't Loop
                // TODO: It should be unified in MemoryManager with memory reuse in the near future
                const auto ma = ov::as_type_ptr<op::MemoryAccess>(parent_node);
                if (ma && ma->is_full_memory_access_op()) {
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
        }
    }
    return modified;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
