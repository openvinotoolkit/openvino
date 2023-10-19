// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/identify_buffers.hpp"

#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/op/brgemm.hpp"
#include "snippets/snippets_isa.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

namespace {
inline size_t index(size_t col_num, size_t row, size_t col) {
    return row * col_num + col;
}
} // namespace

std::vector<bool> IdentifyBuffers::create_adjacency_matrix(const LinearIR& linear_ir, const BufferSet& buffers) const {
    // There are several sync points for adjacency check:
    // 1. Loop because only in Loop we increment pointers. So if some Buffers in the one Loop have conflict
    //    (cannot be inplace: the different ptr increment and data sizes) they are called as adjacent
    // 2. Brgemm because its blocking implementation requires Buffers with unique memory on all inputs and outputs
    const auto size = buffers.size();
    // TODO: Can we use triangular matrix? Need verify using tests
    std::vector<bool> adj(size * size, false);
    for (size_t i = 0; i < size; ++i)
        adj[index(size, i, i)] = true;

    // < ptr_increment, finalization_offset >
    using ShiftPtrParams = std::pair<int64_t, int64_t>;

    auto get_buffer_idx = [&](const std::shared_ptr<op::Buffer>& buffer) {
        const auto iter = std::find(buffers.cbegin(), buffers.cend(), buffer);
        OPENVINO_ASSERT(iter != buffers.cend(), "Buffer wasn't find in Buffer system of Subgraph");
        return std::distance(buffers.cbegin(), iter);
    };

    auto update_adj_matrix = [&](const std::pair<std::shared_ptr<op::Buffer>, ShiftPtrParams>& buffer,
                                 const std::pair<std::shared_ptr<op::Buffer>, ShiftPtrParams>& neighbour_buffer) {
        const bool equal_ptr_params_shifting = buffer.second == neighbour_buffer.second;
        const bool equal_element_type_sizes = buffer.first->get_element_type().size() == neighbour_buffer.first->get_element_type().size();
        if (!equal_ptr_params_shifting || ((buffer.second.first != 0 || buffer.second.second != 0) && !equal_element_type_sizes)) {
            const auto buffer_idx = get_buffer_idx(buffer.first);
            const auto neighbour_idx = get_buffer_idx(neighbour_buffer.first);
            adj[index(size, neighbour_idx, buffer_idx)] = adj[index(size, buffer_idx, neighbour_idx)] = true;
        }
    };

    auto is_buffer = [](const ExpressionPort& port) {
        return ov::is_type<op::Buffer>(port.get_expr()->get_node());
    };

    for (auto expr_it = linear_ir.cbegin(); expr_it != linear_ir.cend(); expr_it++) {
        const auto &expr = *expr_it;
        if (const auto brgemm = ov::as_type_ptr<op::Brgemm>(expr->get_node())) {
            const auto consumers = expr->get_output_port_connector(0)->get_consumers();

            auto buffer_it = std::find_if(consumers.begin(), consumers.end(), is_buffer);
            if (buffer_it == consumers.end())
                continue;
            OPENVINO_ASSERT(std::count_if(consumers.begin(), consumers.end(), is_buffer) == 1, "Brgemm mustn't have more than 1 consumer buffer");

            std::vector<std::shared_ptr<op::Buffer>> adjacency_buffers;
            adjacency_buffers.push_back(ov::as_type_ptr<op::Buffer>(buffer_it->get_expr()->get_node()));

            for (const auto& input_connector : expr->get_input_port_connectors()) {
                const auto parent_node = input_connector->get_source().get_expr()->get_node();
                if (const auto neighbour_buffer = ov::as_type_ptr<op::Buffer>(parent_node)) {
                    adjacency_buffers.push_back(neighbour_buffer);
                }
            }
            for (auto buffer_it = adjacency_buffers.begin(); buffer_it != adjacency_buffers.end(); ++buffer_it) {
                for (auto neighbour_it = std::next(buffer_it); neighbour_it != adjacency_buffers.end(); ++neighbour_it) {
                    const auto buffer_idx = get_buffer_idx(*buffer_it);
                    const auto neighbour_idx = get_buffer_idx(*neighbour_it);
                    adj[index(size, neighbour_idx, buffer_idx)] = adj[index(size, buffer_idx, neighbour_idx)] = true;
                }
            }
            continue;
        }

        const auto& loop_end = ov::as_type_ptr<op::LoopEnd>(expr->get_node());
        if (!loop_end)
            continue;

        const auto input_count = loop_end->get_input_num();
        const auto output_count = loop_end->get_output_num();

        const auto ptr_increments = loop_end->get_ptr_increments();
        const auto finalization_offsets = loop_end->get_finalization_offsets();

        // Buffer -> <ptr increment, finalization_offsets>
        std::map<std::shared_ptr<op::Buffer>, ShiftPtrParams> buffer_neighbours;

        for (size_t i = 0; i < input_count; ++i) {
            const auto& parent_output = expr->get_input_port_connector(i)->get_source().get_expr();
            if (const auto buffer = ov::as_type_ptr<op::Buffer>(parent_output->get_node())) {
                buffer_neighbours[buffer] = { ptr_increments[i], finalization_offsets[i] };
            }
        }
        for (size_t i = 0; i < output_count; ++i) {
            // The consumers of the corresponding Store ops
            const auto index = input_count + i;
            const auto consumer_inputs = expr->get_input_port_connector(index)->get_consumers();
            size_t buffer_count = 0;
            size_t loop_count = 0;
            for (const auto& consumer_input : consumer_inputs) {
                const auto& child_node = consumer_input.get_expr()->get_node();
                if (const auto buffer = ov::as_type_ptr<op::Buffer>(child_node)) {
                    buffer_neighbours[buffer] = { ptr_increments[index], finalization_offsets[index] };
                } else if (ov::is_type<op::LoopEnd>(child_node)) {
                    loop_count++;
                }
            }
            if (buffer_count > 0) {
                OPENVINO_ASSERT((buffer_count == 1) && (buffer_count + loop_count == consumer_inputs.size()),
                                "Loop output must have not more than 1 Buffer");
            }
        }

        for (auto buffer_it = buffer_neighbours.begin(); buffer_it != buffer_neighbours.end(); ++buffer_it) {
            for (auto neighbour_it = std::next(buffer_it); neighbour_it != buffer_neighbours.end(); ++neighbour_it) {
                update_adj_matrix(*buffer_it, *neighbour_it);
            }
        }
    }

    return adj;
}

auto IdentifyBuffers::coloring(BufferSet& buffers, std::vector<bool>& adj) -> std::map<size_t, BufferSet> {
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

bool IdentifyBuffers::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::IdentifyBuffers")
    // Unite Buffers using Graph coloring algorithm.
    // Notes: We identify only Buffer with Intermediate memory because Buffers with new memory are used only in Brgemm case
    //        so these Buffers are always IntermediateBuffer nonadjacent
    BufferSet buffer_exprs;

    for (const auto& expr : linear_ir) {
        if (const auto buffer = ov::as_type_ptr<op::Buffer>(expr->get_node())) {
            buffer_exprs.push_back(buffer);
        }
    }

    // Creation of Adj matrix
    auto adj = create_adjacency_matrix(linear_ir, buffer_exprs);

    // Graph coloring algorithm
    const auto color_groups = coloring(buffer_exprs, adj);

    for (const auto& pair : color_groups) {
        const auto color = pair.first;
        const auto& united_buffers = pair.second;
        for (const auto& buffer : united_buffers) {
            buffer->set_id(color);
        }
    }

    return true;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
