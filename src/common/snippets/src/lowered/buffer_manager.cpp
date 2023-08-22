// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "snippets/lowered/buffer_manager.hpp"

#include "snippets/pass/tokenization.hpp"
#include "snippets/op/buffer.hpp"
#include "snippets/op/memory_access.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/itt.hpp"


namespace ov {
namespace snippets {
namespace lowered {

BufferManager::BufferManager(const lowered::LinearIR& linear_ir) {
    // Initialize edges and nodes
    init_clusters(linear_ir);
}

int64_t BufferManager::allocate() {
    initialization();

    return m_scratchpad_size;
}

void BufferManager::init_clusters(const LinearIR& linear_ir) {
    int64_t order = 0;
    for (const auto& expr : linear_ir) {
        const auto op = expr->get_node();
        if (ov::is_type<ngraph::op::v0::Constant>(op) ||
            ov::is_type<ngraph::op::v0::Parameter>(op) ||
            ov::is_type<ngraph::op::v0::Result>(op) ||
            ov::is_type<op::LoopBegin>(op))
            continue;

        if (const auto buffer = ov::as_type_ptr<op::Buffer>(op)) {
            ov::snippets::pass::SetTopologicalOrder(buffer, order++);
            buffer_clusters.push_back(BufferCluster{expr}); // TODO: Add support of inplace
            continue;
        }
        if (const auto loop_end = ov::as_type_ptr<op::LoopEnd>(op)) {
            // LoopBegin should have the same order as the corresponding LoopEnd
            const auto loop_begin = loop_end->get_loop_begin();
            ov::snippets::pass::SetTopologicalOrder(loop_begin, order);
            ov::snippets::pass::SetTopologicalOrder(op, order++);
            continue;
        }

        //bool is_node = false;  // Meaning in MemoryManager bounds
        //for (size_t i = 0; i < op->get_input_size() && !is_node; ++i) {
        //    is_node = is_node || ov::is_type<op::Buffer>(op->get_input_node_shared_ptr(i));
        //}
        //for (size_t i = 0; i < op->get_output_size() && !is_node; ++i) {
        //    const auto target_consumers = op->get_output_target_inputs(i);
        //    for (const auto& in : target_consumers) {
        //        if (ov::is_type<op::Buffer>(in.get_node())) {
        //            is_node = true;
        //            break;
        //        }
        //    }
        //}

        // if (is_node) {
            // ov::snippets::pass::SetTopologicalOrder(op, order++);
        // }
    }
}


void BufferManager::initialization() {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::BufferManager::initialization")

    size_t buffer_id = 0;
    size_t buffer_offset = 0;

    for (const auto& cluster : buffer_clusters) {
        for (const auto& buffer_expr : cluster) {
            const auto node = buffer_expr->get_node();
            const auto buffer = ov::as_type_ptr<ov::snippets::op::Buffer>(node);
            if (!buffer)
                continue;

            const auto byte_size = buffer->get_byte_size();
            propagate_offset(buffer_expr, buffer_offset);
            buffer->set_id(buffer_id);

            buffer_offset += byte_size;
            buffer_id++;
        }
    }
    m_scratchpad_size = buffer_offset;
}

void BufferManager::propagate_offset(const ExpressionPtr& buffer_expr, const size_t offset) const {
    // If Buffer has offset We set this offset in the connected MemoryAccess ops
    // to correctly read and write data because all Buffers have the common data pointer on buffer scratchpad

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

} // namespace lowered
} // namespace snippets
} // namespace ov
