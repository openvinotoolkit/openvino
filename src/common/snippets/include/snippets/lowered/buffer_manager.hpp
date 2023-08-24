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

private:
    using BufferCluster = std::set<ExpressionPtr>;
    using BufferClusters = std::vector<BufferCluster>;

    /**
     * @brief init Buffers as graph edges and other subgraph around the Buffers as Nodes using enumeration
     *        Parameter
     *    |--- LoopBegin         Parameter
     *    |   LoadReshape         <Edge>   <- already allocated (since after Parameter). Skip
     *    |     Store      --->    Node    <- (LoopBegin,...,LoopEnd)
     *    |--- LoopEnd            <Edge>   <- Buffer. Intermediate memory (edge)
     *          Buffer             Node    <- (LoopBegin,...,LoopEnd)
     *    |--- LoopBegin           ...
     *    |      ...
     */
    void init_clusters(const lowered::LinearIR& linear_ir);
    /**
     * @brief init boxes for MemorySolver
     */
    void init_boxes();
    /**
     * @brief Default initialization: makes non-inplace Buffers and sets different Buffer IDs, offsets
     */
    void initialization();
    /**
     * @brief Set offset to Buffer op and propagate it to the connected memory access ops
     */
    void set_buffer_offset(const ExpressionPtr& buffer_expr, const size_t offset) const;

    BufferClusters buffer_clusters = {};
    std::vector<MemorySolver::Box> boxes = {};
    size_t m_scratchpad_size = 0;
    bool m_enable_optimizations = true;
    constexpr static int64_t alignment = 32; // 32 bytes
};

} // namespace lowered
} // namespace snippets
} // namespace ov
