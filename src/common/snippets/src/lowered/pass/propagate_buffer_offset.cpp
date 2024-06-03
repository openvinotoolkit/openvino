// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "snippets/lowered/pass/propagate_buffer_offset.hpp"

#include "snippets/op/memory_access.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/op/buffer.hpp"
#include "snippets/utils.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {


void PropagateBufferOffset::propagate(const ExpressionPtr& buffer_expr) {
    // If Buffer has offset We set this offset in the connected MemoryAccess ops
    // to correctly read and write data because all Buffers have the common data pointer on buffer scratchpad

    const auto buffer = ov::as_type_ptr<op::Buffer>(buffer_expr->get_node());
    OPENVINO_ASSERT(buffer, "Failed to propagate Buffer offset: PropagateBufferOffset expects Buffer op");
    const auto offset = buffer->get_offset();

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
                    "PropagateBufferOffset didn't find the connected MemoryAccess op to Buffer for offset propagation");
        }
    }
    // Propagate to down: in Load. Buffer can have several Load
    const auto& shape_infer_seq = utils::get_first_child_shape_infer_expr_seq(buffer_expr);
    const auto& target_expr = shape_infer_seq.empty() ? buffer_expr : shape_infer_seq.back();
    const auto& buffer_out = target_expr->get_output_port_connector(0);
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
                "PropagateBufferOffset didn't find the connected MemoryAccess op to Buffer for offset propagation for offset propagation");
        }
    }
}

bool PropagateBufferOffset::run(lowered::LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::PropagateBufferOffset");

    const auto& buffer_expressions = linear_ir.get_buffers();
    for (const auto& buffer_expr : buffer_expressions) {
        propagate(buffer_expr);
    }

    return true;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
