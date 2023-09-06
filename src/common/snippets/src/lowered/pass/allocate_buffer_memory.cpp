// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "snippets/lowered/pass/allocate_buffer_memory.hpp"

#include "snippets/lowered/pass/solve_buffer_memory.hpp"
#include "snippets/lowered/pass/init_buffers.hpp"
#include "snippets/lowered/pass/identify_buffers.hpp"
#include "snippets/lowered/pass/define_buffer_clusters.hpp"
#include "snippets/lowered/pass/clean_repeated_ptr_shifts.hpp"
#include "snippets/pass/tokenization.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

void AllocateBufferMemory::set_buffer_offset(const ExpressionPtr& buffer_expr, const size_t offset) {
    // If Buffer has offset We set this offset in the connected MemoryAccess ops
    // to correctly read and write data because all Buffers have the common data pointer on buffer scratchpad

    const auto buffer = ov::as_type_ptr<op::Buffer>(buffer_expr->get_node());
    OPENVINO_ASSERT(buffer, "Failed to set Buffer offset: AllocateBufferMemory expects Buffer op");
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

bool AllocateBufferMemory::EnumerateExprs::run(lowered::LinearIR& linear_ir) {
    int64_t order = 0;
    for (const auto& expr : linear_ir) {
        ov::snippets::pass::SetTopologicalOrder(expr->get_node(), order++);
    }
    return order > 0;
}

bool AllocateBufferMemory::run(lowered::LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::AllocateBufferMemory");
    if (!is_optimized) {
        auto default_init_pass = InitBuffers();
        default_init_pass.run(linear_ir);
        m_buffer_scratchpad_size = default_init_pass.get_scratchpad_size();
        return  m_buffer_scratchpad_size > 0;
    }

    PassPipeline pipeline;
    pipeline.register_pass<EnumerateExprs>();
    pipeline.register_pass<IdentifyBuffers>();
    pipeline.register_pass<DefineBufferClusters>();
    pipeline.register_pass<CleanRepeatedDataPointerShifts>();
    pipeline.run(linear_ir);

    auto memory_solver_pass = SolveBufferMemory(pipeline.get_pass<DefineBufferClusters>()->get_clusters());
    memory_solver_pass.run(linear_ir);

    m_buffer_scratchpad_size = memory_solver_pass.get_scratchpad_size();

    return m_buffer_scratchpad_size > 0;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
