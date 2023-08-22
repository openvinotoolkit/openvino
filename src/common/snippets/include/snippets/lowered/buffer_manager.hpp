// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "linear_ir.hpp"

#include "memory_solver.hpp"

namespace ov {
namespace snippets {
namespace lowered {

/**
 * @interface MemoryManager
 * @brief Helps to solve issue of optimal memory allocation only for Buffers in graph using MemorySolver
 * @ingroup snippets
 */
class BufferManager {
public:
    BufferManager(const lowered::LinearIR& linear_ir);

    /**
     * @brief allocate optimal memory size using MemorySolver
     * @return size of common memory blob
     */
    int64_t allocate();

    void propagate_offset(const ExpressionPtr& buffer_expr, const size_t offset) const;

private:
    using BufferCluster = std::set<ExpressionPtr>;
    using BufferClusters = std::vector<BufferCluster>;

    /**
     * @brief init Buffers as graph edges and other subgraph around the Buffers as Nodes using enumeration
     *        Parameter
     *    |--- LoopBegin         Parameter
     *    |   LoadReshape         <Edge>   <- already allocated. Skip
     *    |     Store      --->    Node    <- (LoopBegin,...,LoopEnd)
     *    |--- LoopEnd            <Edge>   <- Buffer. Intermediate memory (edge)
     *          Buffer             Node    <- (LoopBegin,...,LoopEnd)
     *    |--- LoopBegin           ...
     *    |      ...
     */
    void init_clusters(const lowered::LinearIR& linear_ir);

    /**
     * @brief Default initialization: makes non-inplace Buffers and sets different Buffer IDs, offsets
     */
    void initialization();

    BufferClusters buffer_clusters;
    size_t m_scratchpad_size = 0;
    bool m_enable_optimizations = true;
};

} // namespace lowered
} // namespace snippets
} // namespace ov
