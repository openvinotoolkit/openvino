// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"

#include "snippets/op/buffer.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface IdentifyBuffers
 * @brief The pass set identifiers for Buffers in common Buffer system.
 *        The buffers with the same identifier will be assigned the same data register.
 *        The pass uses greedy graph coloring algorithm using adjacency matrix:
 *          - Buffers - are vertices of graph;
 *          - Loops, Brgemm (the same other ops) - are "edges" between Buffers (hub of edges).
 *                   The buffers are connected to the same Loop - are adjacent in graph sense bounds.
 *          - The vertices (buffers) are adjacent if they are connected to the same Loop and
 *            their data pointers cannot be proportionally incremented in Loops: different ptr increments or data sizes;
 *          - Firstly, create adjacency matrix using the definition above;
 *          - Secondly, assign the same color to non-adjacent vertices of graph (buffers), and use different colors otherwise.
 *        Note: should be called before ResetBuffer() pass to have correct offsets
 * @ingroup snippets
 */
class IdentifyBuffers: public Pass {
public:
    OPENVINO_RTTI("IdentifyBuffers", "Pass")
    IdentifyBuffers() = default;

    bool run(LinearIR& linear_ir) override;

private:
    using BufferSet = std::vector<std::shared_ptr<op::Buffer>>;

    std::vector<bool> create_adjacency_matrix(const LinearIR& linear_ir, const BufferSet& buffers) const;
    std::map<size_t, BufferSet> coloring(BufferSet& buffers, std::vector<bool>& adj);
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
