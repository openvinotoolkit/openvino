// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/compute_buffer_allocation_size.hpp"

#include "snippets/op/buffer.hpp"
#include "snippets/utils/utils.hpp"
#include "snippets/itt.hpp"


namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

namespace {
std::vector<size_t> get_parent_inner_loops(const std::vector<size_t>& parent_loops, const std::vector<size_t>& current_loops) {
    const auto common_rank = std::min(parent_loops.size(), current_loops.size());
    size_t i = 0;
    while (i < common_rank && parent_loops[i] == current_loops[i])
        ++i;
    return std::vector<size_t>(parent_loops.cbegin() + i, parent_loops.cend());
}
}  // namespace

// Ticket: 113744
// TODO: This logic covers only several specific cases so it should be generalized.
size_t ComputeBufferAllocationSize::get_allocation_size(const LoopManagerPtr& loop_manager, const ExpressionPtr& buffer_expr, size_t allocation_rank) {
    const auto& current_buffer = ov::as_type_ptr<op::Buffer>(buffer_expr->get_node());
    OPENVINO_ASSERT(current_buffer, "`get_allocation_size` expected Buffer");

    // Note: Buffer expressions can have more than one parent after the loops splitting transformation, but only the last parent
    // can be used to access valid loop ports. More info in the ticket: 146646
    const auto buffer_in_idx = buffer_expr->get_input_count() - 1;
    const auto& parent_port = buffer_expr->get_input_port_connector(buffer_in_idx)->get_source();
    const auto& parent_loop_ids = get_parent_inner_loops(parent_port.get_expr()->get_loop_ids(), buffer_expr->get_loop_ids());
    const auto planar_shape = utils::get_preordered_vdims(parent_port);

    const size_t rank = allocation_rank >= 0 ? std::min(static_cast<size_t>(allocation_rank), planar_shape.size())
                                             : planar_shape.size();

    const auto& subtensor =  ov::snippets::utils::get_projected_subtensor(parent_port);

    auto hard_equal = [&parent_port](const LoopPort& port) {
        return *port.expr_port == parent_port;
    };
    auto soft_equal = [&](const LoopPort& loop_port) {
        const auto& port = *loop_port.expr_port;
        // Check semantic of LoopPort
        if (parent_port.get_index() != port.get_index() ||
            port.get_expr()->get_node()->get_type_info() != parent_port.get_expr()->get_node()->get_type_info())
            return false;
        // Check that this LoopPort is connected to the same by semantic Buffer
        const auto consumers = port.get_connected_ports();
        for (const auto& consumer : consumers) {
            if (const auto buffer_consumer = ov::as_type_ptr<op::Buffer>(consumer.get_expr()->get_node())) {
                if (buffer_consumer->get_cluster_id() == current_buffer->get_cluster_id() && consumer.get_index() == buffer_in_idx)
                    return true;
            }
        }
        return false;
    };

    size_t allocation_size = 1;
    std::set<size_t> processed_dim_idxs;
    for (const auto& parent_loop : parent_loop_ids) {
        const auto loop_info = loop_manager->get_loop_info(parent_loop);
        const auto& output_ports = loop_info->get_output_ports();
        auto it = std::find_if(output_ports.begin(), output_ports.end(), hard_equal);
        // [149219] : Try to find original loop port if this LoopInfo is cloned after InsertSpecificIterations
        //            and ports are not mapped on the original ExpressionPorts
        if (it == output_ports.end()) {
            it = std::find_if(output_ports.begin(), output_ports.end(), soft_equal);
            OPENVINO_ASSERT(it != output_ports.end(), "compute_allocation_shape: output port of parent loop can not be found");
        }
        const auto& loop_port = *it;
        const auto& dim_idx = loop_port.dim_idx;
        if (loop_port.is_incremented && dim_idx < rank) {
            if (const auto& unified_loop_info = ov::as_type_ptr<UnifiedLoopInfo>(loop_info))
                allocation_size = utils::dynamic_safe_mul(allocation_size, unified_loop_info->get_work_amount());
            else if (const auto& expanded_loop_info = ov::as_type_ptr<ExpandedLoopInfo>(loop_info))
                allocation_size = utils::dynamic_safe_mul(allocation_size, expanded_loop_info->get_unified_loop_info()->get_work_amount());
            else
                OPENVINO_THROW("Unknown LoopInfo type");
            processed_dim_idxs.insert(dim_idx);
        }
    }
    const auto processing_rank = !processed_dim_idxs.empty() ? std::max(*processed_dim_idxs.rbegin(), subtensor.size()) : subtensor.size();
    for (size_t i = 0; i < std::min(processing_rank, rank); ++i) {
        if (processed_dim_idxs.count(i) == 0) {
            const auto multiplier = i < subtensor.size() ? *(subtensor.rbegin() + i) : *(planar_shape.rbegin() + i);
            allocation_size = utils::dynamic_safe_mul(allocation_size, multiplier);
        }
    }

    // Corner case when the current information is not enough
    if (processing_rank == 0 && processed_dim_idxs.empty()) {
        for (size_t i = 0; i < rank; ++i) {
            allocation_size = utils::dynamic_safe_mul(allocation_size, *(planar_shape.rbegin() + i));
        }
    }

    return allocation_size;
}

bool ComputeBufferAllocationSize::run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::ComputeBufferAllocationSize")

    const auto& loop_manager = linear_ir.get_loop_manager();

    const auto& buffer_expressions = linear_ir.get_buffers();
    for (const auto& buffer_expr : buffer_expressions) {
        const auto node = buffer_expr->get_node();
        if (const auto buffer = ov::as_type_ptr<op::IntermediateMemoryBuffer>(node)) {
            // If the current size is undefined, update it
            // TODO [143395] : MemoryManager will return container with only dynamic buffers without any `is_defined()`
            if (!buffer->is_defined())
                buffer->set_allocation_size(get_allocation_size(loop_manager, buffer_expr, m_buffer_allocation_rank));
        } else {
            OPENVINO_ASSERT(ov::is_type<op::NewMemoryBuffer>(node), "Expected Buffer ops in Buffer expressions of LinearIR");
        }
    }

    return true;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
