// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/solve_buffer_memory.hpp"

#include "snippets/pass/tokenization.hpp"
#include "snippets/utils.hpp"
#include "snippets/itt.hpp"


namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

std::vector<ov::MemorySolver::Box> SolveBufferMemory::init_boxes(const AllocateBuffers::BufferClusters& buffer_clusters) {
    std::vector<ov::MemorySolver::Box> boxes;
    const auto count = static_cast<int>(buffer_clusters.size());
    for (int i = 0; i < count; i++) {
        ov::MemorySolver::Box box = { std::numeric_limits<int>::max(), 0, 0, i };
        int64_t box_size = 0;
        for (const auto& buffer_expr : buffer_clusters[i]) {
            int e_start = 0, e_finish = 0;
            const auto buffer = ov::as_type_ptr<ov::snippets::op::Buffer>(buffer_expr->get_node());
            OPENVINO_ASSERT(buffer != nullptr, "BufferSolver expects Buffer ops in clusters");

            // life finish time - order of LoopEnd / MemoryAccess ops
            const auto& buffer_outs = buffer_expr->get_output_port_connectors();
            for (const auto& buffer_out : buffer_outs) {
                const auto consumers = buffer_out->get_consumers();
                for (const auto& consumer : consumers) {
                    const auto consumer_order = static_cast<int>(ov::snippets::pass::GetTopologicalOrder(consumer.get_expr()->get_node()));
                    e_finish = std::max(e_finish, consumer_order);  // the last consumer
                }
            }
            e_start = e_finish;

            const auto& buffer_ins = buffer_expr->get_input_port_connectors();
            for (const auto& buffer_in : buffer_ins) {
                const auto& source = buffer_in->get_source();
                e_start = static_cast<int>(ov::snippets::pass::GetTopologicalOrder(source.get_expr()->get_node()));

                const auto buffer_siblings = buffer_in->get_consumers();
                for (const auto& sibling : buffer_siblings) {
                    if (const auto loop_end = ov::as_type_ptr<ov::snippets::op::LoopEnd>(sibling.get_expr()->get_node())) {
                        e_start = std::min(e_start, static_cast<int>(ov::snippets::pass::GetTopologicalOrder(loop_end->get_loop_begin())));
                    }
                }
            }
            OPENVINO_ASSERT(e_start <= e_finish, "Incorrect life time of buffer!");

            auto buffer_size = static_cast<int64_t>(buffer->get_byte_size());
            box_size = std::max(buffer_size, box_size);

            box.start = std::min(e_start, box.start);
            box.finish = std::max(e_finish, box.finish);
        }

        // We use data alignment to put data in the line cache
        box.size = utils::div_up(box_size, m_alignment);
        boxes.push_back(box);
    }
    return boxes;
}


bool SolveBufferMemory::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::SolveBufferMemory");

    const auto boxes = init_boxes(m_clusters);

    ov::MemorySolver memSolver(boxes);
    m_buffer_scratchpad_size = static_cast<size_t>(memSolver.solve()) * m_alignment;  // alignment in byte

    // Set offsets for Buffers
    for (const auto& box : boxes) {
        for (const auto& buffer : m_clusters[box.id]) {
            const auto offset = static_cast<size_t>(memSolver.get_offset(static_cast<int>(box.id)));
            AllocateBuffers::set_buffer_offset(buffer, offset * m_alignment);  // alignment in byte
        }
    }
    return m_buffer_scratchpad_size > 0;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
