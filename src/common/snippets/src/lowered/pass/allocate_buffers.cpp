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
    buffer->set_offset(static_cast<int64_t>(offset));

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
    // [113664] The pass contains two main logics: it defines which of buffers can be inplace (use the same memory) and
    // allocate memory of needed size. It should be splitted into several passes and updated in bounds of the ticket 113664.

    // [113664] At the moment New Memory Buffer is used only in BrgemmCPU for AMX case. This memory can be reused for each Brgemm.
    // This plugin-specific condition will be removed in the near future after the task 113664 will be implemented
    size_t offset = 0, new_memory_buffer_offset = 0;
    size_t prev_data_size = 0, current_data_size = 0;
    std::set<ExpressionPtr> allocated_buffers;
    bool new_memory_buffer_allocated = false;

    auto allocate = [&](const std::shared_ptr<op::Buffer>& buffer, const ExpressionPtr& expr, size_t buffer_size) {
        offset = m_buffer_scratchpad_size;
        propagate_offset(linear_ir, expr, offset);
        m_buffer_scratchpad_size += buffer_size;
        allocated_buffers.insert(expr);
        prev_data_size = current_data_size;
    };

    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        const auto& expr = *expr_it;
        if (auto buffer = as_type_ptr<op::Buffer>(expr->get_node())) {
            const auto buffer_size = buffer->get_byte_size();
            current_data_size = buffer->get_element_type().size();
            // If it's the first buffer, offsets are zero => nothing to propagate, can continue
            if (m_buffer_scratchpad_size == 0) {
                m_buffer_scratchpad_size += buffer_size;
                allocated_buffers.insert(expr);
                prev_data_size = current_data_size;
                continue;
            }

            if (buffer->is_intermediate_memory()) {
                const auto& parent_expr = expr->get_input_port_connector(0)->get_source().get_expr();
                const auto& parent_node = parent_expr->get_node();
                // Full MemoryAccess ops need new memory. Previous logic is to check for parent isn't Loop
                // [113664] It should be unified in MemoryManager with memory reuse in the near future
                const auto ma = ov::as_type_ptr<op::MemoryAccess>(parent_node);
                if (ma && ma->is_full_memory_access_op()) {
                    allocate(buffer, *expr_it, buffer_size);
                    continue;
                }

                // Loop       Full_MA
                //  |           |
                // Buffer_1  Buffer_0
                //   \         /
                //     Full_MA
                // At the moment the pass support only sequentially implicit InPlace.
                // If Buffer_0 is allocated firstly as Buffer after full memory access op,
                // we cannot reuse this allocated memory for Buffer_1 - we must allocate new memory for it.
                // [113664] It should be unified in MemoryManager with memory reuse in the near future
                bool need_allocate = false;
                const auto consumers = expr->get_output_port_connector(0)->get_consumers();
                for (const auto& consumer : consumers) {
                    const auto& consumer_expr = consumer.get_expr();
                    const auto& child_node = consumer_expr->get_node();
                    const auto ma = ov::as_type_ptr<op::MemoryAccess>(child_node);
                    if (ma && ma->is_full_memory_access_op()) {
                        for (size_t i = 0; i < consumer_expr->get_input_count() && !need_allocate; ++i) {
                            if (i == consumer.get_index())
                                continue;
                            const auto buffer_sibling = consumer_expr->get_input_port_connector(i)->get_source().get_expr();
                            need_allocate = ov::is_type<op::Buffer>(buffer_sibling->get_node()) && allocated_buffers.count(buffer_sibling) != 0;
                        }
                    }
                    if (need_allocate)
                        break;
                }
                if (need_allocate) {
                    allocate(buffer, *expr_it, buffer_size);
                    continue;
                }

                // [113664] For more details and reason of the current solution, please, go to the ticket description
                const auto current_allocated_memory_size = m_buffer_scratchpad_size - offset;
                if (((current_data_size == prev_data_size) && buffer_size > current_allocated_memory_size) ||
                    ((current_data_size != prev_data_size) && buffer_size != current_allocated_memory_size)) {
                    allocate(buffer, expr, buffer_size);
                    continue;
                }
                propagate_offset(linear_ir, *expr_it, offset);
                allocated_buffers.insert(expr);
                prev_data_size = current_data_size;
            } else {
                if (!new_memory_buffer_allocated) {
                    allocate(buffer, *expr_it, buffer_size);
                    new_memory_buffer_allocated = true;
                    new_memory_buffer_offset = offset;
                } else {
                    propagate_offset(linear_ir, *expr_it, new_memory_buffer_offset);
                    allocated_buffers.insert(expr);
                    prev_data_size = current_data_size;
                }
            }
        }
    }
    return !allocated_buffers.empty();
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
