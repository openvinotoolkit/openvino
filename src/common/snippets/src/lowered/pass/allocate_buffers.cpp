// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "snippets/lowered/pass/allocate_buffers.hpp"

#include "snippets/lowered/pass/enumerate_expressions.hpp"
#include "snippets/lowered/pass/solve_buffer_memory.hpp"
#include "snippets/lowered/pass/init_buffers_default.hpp"
#include "snippets/lowered/pass/identify_buffers.hpp"
#include "snippets/lowered/pass/define_buffer_clusters.hpp"
#include "snippets/lowered/pass/normalize_buffer_ids.hpp"
#include "snippets/pass/tokenization.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

AllocateBuffers::AllocateBuffers(size_t& buffer_scratchpad_size, bool is_optimized)
    : m_buffer_scratchpad_size(buffer_scratchpad_size), m_is_optimized_mode(is_optimized) {}

void AllocateBuffers::set_buffer_offset(const ExpressionPtr& buffer_expr, const size_t offset) {
    // If Buffer has offset We set this offset in the connected MemoryAccess ops
    // to correctly read and write data because all Buffers have the common data pointer on buffer scratchpad

    const auto buffer = ov::as_type_ptr<op::Buffer>(buffer_expr->get_node());
    OPENVINO_ASSERT(buffer, "Failed to set Buffer offset: AllocateBuffers expects Buffer op");
    buffer->set_offset(static_cast<int64_t>(offset));

    // Propagate to up: in Store. Buffer can have only one Store
    if (ov::is_type<op::IntermediateMemoryBuffer>(buffer)) {
        OPENVINO_ASSERT(buffer_expr->get_input_port_connectors().size() == 1, "Buffer with intermediate memory must have one parent");
        const auto& parent_output = buffer_expr->get_input_port_connector(0)->get_source();
        const auto& parent_expr = parent_output.get_expr();
        const auto port = parent_output.get_index();
        const auto& parent_node = parent_expr->get_node();
        auto memory_access = std::dynamic_pointer_cast<modifier::MemoryAccess>(parent_node);
        if (memory_access && memory_access->is_memory_access_output_port(port)) {
            memory_access->set_output_offset(offset, port);
        } else {
            OPENVINO_THROW(
                    "Buffer::set_offset() was called when Buffer didn't have the corresponding MemoryAccess op for offset propagation");
        }
    }
    // Propagate to down: in Load. Buffer can have several Load
    const auto& buffer_out = buffer_expr->get_output_port_connector(0);
    for (const auto& child_expr_input : buffer_out->get_consumers()) {
        const auto& child_expr = child_expr_input.get_expr();
        const auto port = child_expr_input.get_index();
        const auto& child_node = child_expr->get_node();
        auto memory_access = std::dynamic_pointer_cast<modifier::MemoryAccess>(child_node);
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

bool AllocateBuffers::run(lowered::LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::AllocateBuffers");
    m_buffer_scratchpad_size = 0;

    if (m_is_optimized_mode) {
        BufferClusters buffer_clusters;
        PassPipeline pipeline;
        pipeline.register_pass<EnumerateExpressions>();
        pipeline.register_pass<IdentifyBuffers>();
        pipeline.register_pass<DefineBufferClusters>(buffer_clusters);
        pipeline.register_pass<SolveBufferMemory>(m_buffer_scratchpad_size, buffer_clusters);
        pipeline.register_pass<NormalizeBufferIDs>();
        pipeline.run(linear_ir);
    } else {
        InitBuffersDefault(m_buffer_scratchpad_size).run(linear_ir, linear_ir.cbegin(), linear_ir.cend());
    }

    return m_buffer_scratchpad_size > 0;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
