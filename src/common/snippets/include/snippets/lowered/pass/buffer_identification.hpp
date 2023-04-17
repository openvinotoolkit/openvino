// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "transformation.hpp"

namespace ngraph {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface BufferIdentification
 * @brief The pass set identifiers for Buffers in common Buffer system.
 *        The buffers with the same identifier has the same data register.
 *        The pass uses greedy graph coloring algorithm using adjacency matrix:
 *          - Buffers - are vertices of graph
 *          - Loops, Brgemm (the same other ops) - are "edges" between Buffers (hub of edges).
 *                   The buffers are connected to the same Loop - are adjacent in graph sense bounds.
 *          - The vertices (buffers) are adjacent if they are connected to the same Loop and
 *            their data pointers cannot be proportionally incremented in Loops: different ptr increments or data sizes.
 *          - Firstly, create adjacency matrix using the definition above
 *          - Secondly, color vertices of graph (buffers) using adjacency matrix
 *        Note: should be called before ResetBuffer() pass to have correct offsets
 * @ingroup snippets
 */
class BufferIdentification: public Transformation {
public:
    OPENVINO_RTTI("BufferIdentification", "Transformation")
    BufferIdentification() = default;

    bool run(LinearIR& linear_ir) override;

private:
    using BufferSet = std::vector<ExpressionPtr>;

    std::vector<bool> create_adjacency_matrix(const LinearIR& linear_ir, const BufferSet& buffers) const;
    std::map<size_t, BufferSet> coloring(BufferSet& buffers, std::vector<bool>& adj);
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ngraph
