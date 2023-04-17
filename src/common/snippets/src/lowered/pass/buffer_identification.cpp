// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/buffer_identification.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

namespace ngraph {
namespace snippets {
namespace lowered {
namespace pass {

namespace {
auto is_intermediate_buffer(const std::shared_ptr<ov::Node>& op) -> std::shared_ptr<op::Buffer> {
    const auto buffer = ov::as_type_ptr<op::Buffer>(op);
    return buffer && buffer->is_intermediate_memory() ? buffer : nullptr;
}

inline size_t index(size_t col_num, size_t row, size_t col) {
    return row * col_num + col;
}
} // namespace

std::vector<bool> BufferIdentification::create_adjacency_matrix(const LinearIR& linear_ir, const BufferSet& buffers) const {
    // The sync point to check for adjacency is Loop because only in Loop we increment pointers.
    // So if some Buffers in the one Loop have conflict (cannot be inplace: the different ptr increment and data sizes)
    // they are called as adjacent
    const auto size = buffers.size();
    // TODO: Can we use triangular matrix? Need verify using tests
    std::vector<bool> adj(size * size, false);
    for (size_t i = 0; i < size; ++i)
        adj[index(size, i, i)] = true;

    auto update_adj_matrix = [&](const std::shared_ptr<op::Buffer>& buffer, size_t buffer_index,
                                 const std::shared_ptr<op::Buffer>& neighbour_buffer,
                                 size_t buffer_loop_port, size_t neighbour_buffer_loop_port,
                                 const std::vector<int64_t>& ptr_increments,
                                 const std::vector<int64_t>& io_data_sizes) {
        if (neighbour_buffer) {
            // TODO: What's about finalization offsets? It's needed?
            if (ptr_increments[buffer_loop_port] != ptr_increments[neighbour_buffer_loop_port] ||
                io_data_sizes[buffer_loop_port] != io_data_sizes[neighbour_buffer_loop_port]) {
                const auto iter = std::find(buffers.cbegin(), buffers.cend(), linear_ir.get_expr_by_node(neighbour_buffer));
                NGRAPH_CHECK(iter != buffers.cend(), "Buffer wasn't find in Buffer system of Subgraph");

                const size_t adj_idx = std::distance(buffers.cbegin(), iter);
                adj[index(size, adj_idx, buffer_index)] = adj[index(size, buffer_index, adj_idx)] = true;
            }
        }
    };

    for (size_t buffer_idx = 0; buffer_idx < buffers.size(); ++buffer_idx) {
        // Here intermediate Buffer
        const auto buffer_expr = buffers[buffer_idx];
        const auto buffer_input_tds = buffer_expr->get_inputs();
        OPENVINO_ASSERT(buffer_input_tds.size() == 1, "Intermediate Buffer must have one input");
        const auto buffer = ov::as_type_ptr<op::Buffer>(buffer_expr->get_node());

        const auto& buffer_td = buffer_input_tds.front();
        const auto buffer_siblings = linear_ir.get_exprs_by_input(buffer_td);
        for (const auto& buffer_sibling : buffer_siblings) {
            const auto& sibling_expr = buffer_sibling.expr;
            // Skip myself
            if (sibling_expr == buffer_expr) {
                continue;
            } else if (const auto loop_end = ov::as_type_ptr<op::LoopEnd>(sibling_expr->get_node())) {
                const auto& loop_tds = sibling_expr->get_inputs();
                const auto input_count = loop_end->get_input_num();
                const auto output_count = loop_end->get_output_num();
                const auto& ptr_increments = loop_end->get_ptr_increments();
                const auto& io_data_sizes = loop_end->get_element_type_sizes();
                const auto buffer_loop_port = std::distance(loop_tds.begin(), std::find(loop_tds.begin(), loop_tds.end(), buffer_td));

                // Verify Buffers on Loop inputs:
                for (size_t input_idx = 0; input_idx < input_count; ++input_idx) {
                    const auto loop_in = linear_ir.get_expr_by_output(loop_tds[input_idx]).expr;
                    if (const auto& neighbour_buffer = is_intermediate_buffer(loop_in->get_node())) {
                        const auto neighbour_buffer_loop_port = input_idx;
                        update_adj_matrix(buffer, buffer_idx, neighbour_buffer,
                                          buffer_loop_port, neighbour_buffer_loop_port,
                                          ptr_increments, io_data_sizes);
                    }
                }

                // Verify Buffers on Loop outputs
                for (size_t output_idx = 0; output_idx < output_count; ++output_idx) {
                    // Skip the current Buffer
                    if (buffer_td == loop_tds[input_count + output_idx])
                        continue;

                    const auto& consumer_inputs = linear_ir.get_exprs_by_input(loop_tds[input_count + output_idx]);
                    for (const auto& consumer_input : consumer_inputs) {
                        const auto& child_node = consumer_input.expr->get_node();
                        if (const auto& neighbour_buffer = is_intermediate_buffer(child_node)) {
                            const auto neighbour_buffer_loop_port = input_count + output_idx;
                            update_adj_matrix(buffer, buffer_idx, neighbour_buffer,
                                              buffer_loop_port, neighbour_buffer_loop_port,
                                              ptr_increments, io_data_sizes);
                        }
                    }
                }
            } else {
                throw ov::Exception("Buffer has incorrect siblings! There can be only LoopEnds");
            }
        }
    }

    return adj;
}

auto BufferIdentification::coloring(BufferSet& buffers, std::vector<bool>& adj) -> std::map<size_t, BufferSet> {
    size_t color = 0;
    std::map<size_t, BufferSet> color_groups;
    const auto size = buffers.size();
    for (size_t i = 0; i < size; i++) {
        // The Buffer is already colored (visited) - skip
        if (!buffers[i])
            continue;

        const auto& buffer = buffers[i];
        color_groups[color].push_back(buffer); // Add to Color Group
        buffers[i] = nullptr;  // Remove from graph vertices

        // While Buffer `i` has non-coloured non-neighbours (while row `i` contains 0)
        while (!std::accumulate(adj.begin() + i * size, adj.begin() + (i + 1) * size, true, std::logical_and<bool>())) {
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

bool BufferIdentification::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::BufferIdentification")
    // Unite Buffers using Graph coloring algorithm.
    // Notes: We identify only Buffer with Intermediate memory because Buffers with new memory are used only in Brgemm case
    //        so these Buffers are always IntermediateBuffer nonadjacent
    BufferSet buffer_exprs;

    for (const auto& expr : linear_ir) {
        const auto& op = expr->get_node();
        if (const auto buffer = is_intermediate_buffer(op)) {
            buffer_exprs.push_back(expr);
        }
    }

    // Creation of Adj matrix
    auto adj = create_adjacency_matrix(linear_ir, buffer_exprs);

    // Graph coloring algorithm
    const auto color_groups = coloring(buffer_exprs, adj);

    // FIXME: use const auto& [color, united_buffers] when C++17 is available
    for (const auto& pair : color_groups) {
        const auto color = pair.first;
        const auto& united_buffers = pair.second;
        for (const auto& buffer_expr : united_buffers) {
            const auto buffer = ov::as_type_ptr<op::Buffer>(buffer_expr->get_node());
            buffer->set_id(color);
        }
    }

    return true;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ngraph
