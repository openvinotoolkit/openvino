// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "snippets/lowered/buffer_manager.hpp"

#include "snippets/pass/tokenization.hpp"
#include "snippets/op/buffer.hpp"
#include "snippets/op/memory_access.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/itt.hpp"
#include "snippets/utils.hpp"


namespace ov {
namespace snippets {
namespace lowered {

int64_t BufferManager::allocate(const lowered::LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::BufferManager::allocate")
    const auto buffer_system = create_buffer_system(linear_ir);
    auto scratchpad_size = init_default(buffer_system);

    if (m_enable_optimizations) {
        const auto buffer_clusters = init_clusters(buffer_system);
        const auto boxes = init_boxes(buffer_clusters);

        MemorySolver staticMemSolver(boxes);
        scratchpad_size = static_cast<size_t>(staticMemSolver.solve()) * m_alignment;  // alignment in byte

        // Set offsets for Buffers
        for (auto& box : boxes) {
            for (auto& buffer : buffer_clusters[box.id]) {
                int64_t offset = staticMemSolver.getOffset(box.id);
                set_buffer_offset(buffer, offset * m_alignment);  // alignment in byte
            }
        }
    }
    return scratchpad_size;
}

BufferManager::BufferSystem BufferManager::create_buffer_system(const lowered::LinearIR& linear_ir) {
    int64_t order = 0;
    BufferSystem system;
    for (const auto& expr : linear_ir) {
        const auto op = expr->get_node();
        if (const auto buffer = ov::as_type_ptr<op::Buffer>(op)) {
            system.push_back(expr);
        }
        ov::snippets::pass::SetTopologicalOrder(op, order++);
    }
    return system;
}

size_t BufferManager::init_default(const BufferSystem& buffer_system) {
    size_t buffer_id = 0;
    size_t buffer_offset = 0;
    for (const auto& buffer_expr : buffer_system) {
        const auto node = buffer_expr->get_node();
        const auto buffer = ov::as_type_ptr<ov::snippets::op::Buffer>(node);
        if (!buffer)
            continue;

        const auto byte_size = buffer->get_byte_size();
        set_buffer_offset(buffer_expr, buffer_offset);
        buffer->set_id(buffer_id);

        buffer_offset += byte_size;
        buffer_id++;
    }
    return buffer_offset;
}

BufferManager::BufferClusters BufferManager::init_clusters(const BufferSystem& buffer_system) {
    // TODO: Add support of inplace
    BufferClusters buffer_clusters;
    for (const auto& buffer_expr : buffer_system) {
        buffer_clusters.push_back(BufferCluster{buffer_expr});
    }
    return buffer_clusters;
}

std::vector<MemorySolver::Box> BufferManager::init_boxes(const BufferClusters& buffer_clusters) {
    std::vector<MemorySolver::Box> boxes;
    const auto count = static_cast<int>(buffer_clusters.size());
    for (int i = 0; i < count; i++) {
        MemorySolver::Box box = { std::numeric_limits<int>::max(), 0, 0, i };
        int64_t box_size = 0;
        for (const auto& buffer_expr : buffer_clusters[i]) {
            int e_start = 0, e_finish = 0;
            const auto buffer = ov::as_type_ptr<ov::snippets::op::Buffer>(buffer_expr->get_node());
            OPENVINO_ASSERT(buffer != nullptr, "BufferManager expects Buffer ops in clusters");
            const auto buffer_order = static_cast<int>(ov::snippets::pass::GetTopologicalOrder(buffer));

            // life finish time - order of LoopEnd / MemoryAccess ops
            const auto buffer_outs = buffer_expr->get_output_port_connectors();
            for (const auto& buffer_out : buffer_outs) {
                const auto consumers = buffer_out->get_consumers();
                for (const auto& consumer : consumers) {
                    const auto consumer_order = static_cast<int>(ov::snippets::pass::GetTopologicalOrder(consumer.get_expr()->get_node()));
                    e_finish = std::max(e_finish, consumer_order);
                }
            }
            e_start = e_finish;

            const auto buffer_ins = buffer_expr->get_input_port_connectors();
            for (const auto& buffer_in : buffer_ins) {
                const auto& source = buffer_in->get_source();
                auto local_order = static_cast<int>(ov::snippets::pass::GetTopologicalOrder(source.get_expr()->get_node()));

                const auto buffer_siblings = buffer_in->get_consumers();
                for (const auto& sibling : buffer_siblings) {
                    const auto loop_end = ov::as_type_ptr<ov::snippets::op::LoopEnd>(sibling.get_expr()->get_node());
                    if (!loop_end)
                        continue;
                    const auto loop_end_order = static_cast<int>(ov::snippets::pass::GetTopologicalOrder(loop_end));
                    if (loop_end_order < buffer_order) {
                        local_order = std::min(local_order, static_cast<int>(ov::snippets::pass::GetTopologicalOrder(loop_end->get_loop_begin())));
                    }
                }
                e_start = std::min(e_start, local_order);
            }

            // TODO: Added support of Dynamic Buffers
            auto buffer_size = static_cast<int64_t>(buffer->get_byte_size());
            box_size = std::max(buffer_size, box_size);

            box.start = std::min(e_start, box.start);
            box.finish = std::max(e_finish, box.finish);
        }

        box.size = utils::div_up(box_size, m_alignment);
        boxes.push_back(box);
    }
    return boxes;
}

void BufferManager::set_buffer_offset(const ExpressionPtr& buffer_expr, const size_t offset) {
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
