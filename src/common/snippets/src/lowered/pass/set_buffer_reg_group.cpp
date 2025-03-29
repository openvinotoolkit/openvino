// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/set_buffer_reg_group.hpp"

#include "snippets/lowered/pass/mark_invariant_shape_path.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/lowered/expressions/buffer_expression.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

namespace {
inline size_t index(size_t col_num, size_t row, size_t col) {
    return row * col_num + col;
}
} // namespace

size_t SetBufferRegGroup::get_buffer_idx(const BufferExpressionPtr& target, const BufferPool& pool) {
    const auto iter = std::find(pool.cbegin(), pool.cend(), target);
    OPENVINO_ASSERT(iter != pool.cend(), "Buffer wasn't find in Buffer system of Subgraph");
    return std::distance(pool.cbegin(), iter);
}

bool SetBufferRegGroup::can_be_in_one_reg_group(const UnifiedLoopInfo::LoopPortInfo& lhs_info,
                                                const UnifiedLoopInfo::LoopPortInfo& rhs_info) {
    const auto equal_element_type_sizes = lhs_info.desc.data_size == rhs_info.desc.data_size;
    OPENVINO_ASSERT(lhs_info.port.get_expr_port() && rhs_info.port.get_expr_port(), "Expression ports are nullptr!");
    const auto equal_invariant_shape_paths =
        MarkInvariantShapePath::getInvariantPortShapePath(*lhs_info.port.get_expr_port()) ==
        MarkInvariantShapePath::getInvariantPortShapePath(*rhs_info.port.get_expr_port());
    const auto lhs_is_incremented = lhs_info.port.is_incremented();
    const auto rhs_is_incremented = rhs_info.port.is_incremented();
    const auto equal_is_incremented = lhs_is_incremented == rhs_is_incremented;
    return equal_invariant_shape_paths && equal_is_incremented &&
           (equal_element_type_sizes || !lhs_is_incremented || (lhs_info.desc.ptr_increment == 0 && lhs_info.desc.finalization_offset == 0));
}

bool SetBufferRegGroup::are_adjacent(const BufferMap::value_type& lhs, const BufferMap::value_type& rhs) {
    const auto& lhs_ids = lhs.first->get_loop_ids();
    const auto& rhs_ids = rhs.first->get_loop_ids();
    const auto equal_loop_ids = lhs_ids == rhs_ids;
    if (equal_loop_ids) {  // Buffers are connected to the same Loop and have the same outer Loops
        return !can_be_in_one_reg_group(lhs.second, rhs.second);
    } else {  // Buffers are connected to the same Loop, but one of Buffers - inside this Loop, another - outside
        // Buffers are adjacent if outer Buffer has non-zero data shift params
        if (lhs_ids.size() == rhs_ids.size()) // If the count of outer Loops are equal, it means that outer loops are already different
            return true;
        const auto& outer_buffer = lhs_ids.size() < rhs_ids.size() ? lhs : rhs;
        const auto count_outer_loops = std::min(lhs_ids.size(), rhs_ids.size());
        const auto are_outer_loops_the_same = lhs_ids.size() != rhs_ids.size() &&
            std::equal(rhs_ids.cbegin(), rhs_ids.cbegin() + count_outer_loops, lhs_ids.cbegin());
        const auto outer_buffer_has_zero_shifts = outer_buffer.second.desc.ptr_increment == 0 && outer_buffer.second.desc.finalization_offset == 0;
        return !(are_outer_loops_the_same && outer_buffer_has_zero_shifts);
    }
}

void SetBufferRegGroup::update_adj_matrix(const BufferMap::value_type& lhs, const BufferMap::value_type& rhs, const BufferPool& buffers,
                                          std::vector<bool>& adj) {
    const auto size = buffers.size();
    const auto lhs_idx = get_buffer_idx(lhs.first, buffers);
    const auto rhs_idx = get_buffer_idx(rhs.first, buffers);
    // Already adjacent - skip
    if (adj[index(size, rhs_idx, lhs_idx)])
        return;

    if (are_adjacent(lhs, rhs)) {
        adj[index(size, rhs_idx, lhs_idx)] = adj[index(size, lhs_idx, rhs_idx)] = true;
    }
}

std::vector<bool> SetBufferRegGroup::create_adjacency_matrix(const LoopManagerPtr& loop_manager, LinearIR::constExprIt begin, LinearIR::constExprIt end,
                                                             const BufferPool& pool) {
    // The sync point to check for adjacency is Loop because only in Loop we increment pointers.
    // So if some Buffers in the one Loop have conflict (cannot be inplace: the different ptr increment and data sizes)
    // they are called as adjacent
    const auto size = pool.size();
    std::vector<bool> adj(size * size, false);
    for (size_t i = 0; i < size; ++i)
        adj[index(size, i, i)] = true;

    for (auto expr_it = begin; expr_it != end; expr_it++) {
        const auto &expr = *expr_it;
        const auto& loop_end = ov::as_type_ptr<op::LoopEnd>(expr->get_node());
        if (!loop_end)
            continue;

        const auto& loop_info = loop_manager->get_loop_info<UnifiedLoopInfo>(loop_end->get_id());
        const auto buffer_loop_neighbours = get_buffer_loop_neighbours(loop_info);
        const auto buffers_loop_inside = get_buffer_loop_inside(expr_it);
        for (auto buffer_it = buffer_loop_neighbours.cbegin(); buffer_it != buffer_loop_neighbours.cend(); ++buffer_it) {
            // If Buffers, that are connected to the same Loop, have not proportionally ptr shift params for this Loop - these Buffers are adjacent
            for (auto neighbour_it = std::next(buffer_it); neighbour_it != buffer_loop_neighbours.cend(); ++neighbour_it) {
                update_adj_matrix(*buffer_it, *neighbour_it, pool, adj);
            }
            // Buffers which are connected to the current Loop with zero ptr shifts and Buffers which are inside this Loop - must be adjacent:
            // after each the Loop iteration GPR will be shifted using ptr increment of Buffer outside.
            // But if inner Buffers have the same GPR - it means that these Buffers will work with shifted memory.
            for (auto inner_it = buffers_loop_inside.cbegin(); inner_it != buffers_loop_inside.cend(); ++inner_it) {
                update_adj_matrix(*buffer_it, *inner_it, pool, adj);
            }
        }
    }

    return adj;
}

SetBufferRegGroup::BufferMap SetBufferRegGroup::get_buffer_loop_neighbours(const UnifiedLoopInfoPtr& loop_info) {
    BufferMap buffer_neighbours;

    const auto& loop_inputs = loop_info->get_input_ports_info();
    for (const auto& port_info : loop_inputs) {
        const auto& parent_output = port_info.port.get_expr_port()->get_port_connector_ptr()->get_source().get_expr();
        if (const auto buffer_expr = ov::as_type_ptr<BufferExpression>(parent_output)) {
            if (buffer_neighbours.count(buffer_expr) > 0) {
                const auto& port_desc = port_info.desc;
                OPENVINO_ASSERT(buffer_neighbours[buffer_expr].desc == port_desc,
                                "Invalid data pointer shifts: If Buffer has several consumers, this consumers must have the same shifts or zero");
                continue;
            }
            buffer_neighbours[buffer_expr] = port_info;
        }
    }

    const auto& loop_outputs = loop_info->get_output_ports_info();
    for (const auto& port_info : loop_outputs) {
        const auto& consumer_inputs = port_info.port.get_expr_port()->get_port_connector_ptr()->get_consumers();
        for (const auto& consumer_input : consumer_inputs) {
            const auto& child_expr = consumer_input.get_expr();
            if (const auto buffer_expr = ov::as_type_ptr<BufferExpression>(child_expr))
                buffer_neighbours[buffer_expr] = port_info;
        }
    }

    return buffer_neighbours;
}

SetBufferRegGroup::BufferMap SetBufferRegGroup::get_buffer_loop_inside(const LinearIR::constExprIt& loop_end_it) {
    const auto& loop_end = ov::as_type_ptr<op::LoopEnd>((*loop_end_it)->get_node());
    const auto loop_begin = loop_end->get_loop_begin();
    BufferMap inner_buffers;
    for (auto it = std::reverse_iterator<LinearIR::constExprIt>(loop_end_it); (*it)->get_node() != loop_begin; ++it) {
        const auto& inner_expr = *it;
        if (const auto buffer_expr = ov::as_type_ptr<BufferExpression>(inner_expr)) {
            // Set default value (zeroes) since it's not used for adjacency definition in case with Buffers in Loop
            if (inner_buffers.count(buffer_expr) == 0)
                inner_buffers[buffer_expr] = UnifiedLoopInfo::LoopPortInfo();
        }
    }
    return inner_buffers;
}

auto SetBufferRegGroup::coloring(BufferPool& buffers, std::vector<bool>& adj) -> std::map<size_t, BufferPool> {
    size_t color = 0;
    std::map<size_t, BufferPool> color_groups;
    const auto size = buffers.size();
    for (size_t i = 0; i < size; ++i) {
        // The Buffer is already colored (visited) - skip
        if (!buffers[i])
            continue;

        const auto& buffer = buffers[i];
        color_groups[color].push_back(buffer); // Add to Color Group
        buffers[i] = nullptr;  // Remove from graph vertices

        // While Buffer `i` has non-coloured non-neighbours (while row `i` contains 0)
        while ((i + 1 < size) && !std::accumulate(adj.begin() + i * size, adj.begin() + (i + 1) * size, true, std::logical_and<bool>())) {
            size_t j = i + 1;
            // Find first non-adjacent and non-visited (non-colored) Buffer to color him to the same color
            for (; j < size; ++j) {
                if (!adj[index(size, i, j)] && buffers[j])
                    break;
            }

            // If we don't have the corresponding non-adjacent and non-colored Buffers,
            // we should make break - all potential Buffers for the current color are already colored
            if (j == size)
                break;

            const auto& neighbour_buffer = buffers[j];
            color_groups[color].push_back(neighbour_buffer); // Add to Color Group
            buffers[j] = nullptr;  // Remove from graph vertices
            // Unite adjacency links:
            //    All the neighbors of Buffer `j` are added to the neighbors of Buffer `i` (the `vertices` are pulled together).
            //    The result is an updated i-th row of the adjacency matrix,
            //    in which 0 are only in columns with `vertex` numbers that are not adjacent to either the i-th or j-th `vertices`.
            //    Mathematically, this can be replaced by the operation of OR of Boolean vectors representing strings i and j.
            std::transform(adj.begin() + i * size, adj.begin() + (i + 1) * size, adj.begin() + j * size,
                           adj.begin() + i * size, std::logical_or<bool>());
        }

        color++;
    }

    return color_groups;
}

bool SetBufferRegGroup::run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::SetBufferRegGroup")

    // Identify Buffers using Graph coloring algorithm.
    BufferPool buffer_pool = linear_ir.get_buffers();
    // For the better coloring Buffers should be stored in the order of execution numbers
    std::sort(buffer_pool.begin(), buffer_pool.end(),
              [](const BufferExpressionPtr& lhs, const BufferExpressionPtr& rhs) { return lhs->get_exec_num() < rhs->get_exec_num(); });

    // Creation of Adj matrix
    auto adj = create_adjacency_matrix(linear_ir.get_loop_manager(), begin, end, buffer_pool);

    // Graph coloring algorithm
    const auto color_groups = coloring(buffer_pool, adj);

    for (const auto& pair : color_groups) {
        const auto color = pair.first;
        const auto& united_buffers = pair.second;
        for (const auto& buffer_expr : united_buffers)
            buffer_expr->set_reg_group(color);
    }

    return true;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
