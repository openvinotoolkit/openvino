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
 * @interface BufferSolver
 * @brief Helps to solve issue of optimal memory allocation only for Buffers in graph using the following optimizations:
 *         - MemorySolver: helps to solve issue of optimal memory allocation;
 *         - InPlace: Loop or MemoryAccess ops read from the memory and store data to the same memory if possible
 *         - Reusing Buffer IDs: Buffers have the same IDs (gpr) in cases when Buffers aren't connected or has the same data ptr shifts
 * @ingroup snippets
 */
class BufferSolver {
public:
    BufferSolver() = default;

    int64_t solve(lowered::LinearIR& linear_ir);

private:
    using BufferCluster = std::set<ExpressionPtr>;
    using BufferClusters = std::vector<BufferCluster>;

    void enumerate(const lowered::LinearIR& linear_ir);

    void identify_buffers(lowered::LinearIR& linear_ir);

    BufferClusters init_clusters(const lowered::LinearIR& linear_ir);
    BufferClusters init_default_clusters(const lowered::LinearIR& linear_ir);
    BufferClusters init_inplace_clusters(const lowered::LinearIR& linear_ir);

    int64_t allocate(const BufferClusters& buffer_clusters);
    std::vector<MemorySolver::Box> init_boxes(const BufferClusters& buffer_clusters);

    static void set_buffer_offset(const ExpressionPtr& buffer_expr, const size_t offset);

    enum OptimizationsBit : unsigned {
        DefaultBit = 1lu << 0,          // No optimizations
        MemorySolverBit = 1u << 1,      // MemorySolver using
        InPlaceOneLevelBit = 1u << 2,   // InPlace mechanism on the same level using
        InPlaceMultiLevelBit = 1u << 3, // InPlace mechanism on the different level using
        ReusingBufferIDBit = 1u << 4,     // Reusing Buffer IDs
    };

    unsigned m_mode = OptimizationsBit::MemorySolverBit | OptimizationsBit::ReusingBufferIDBit | OptimizationsBit::InPlaceMultiLevelBit;
    constexpr static int64_t m_alignment = 32; // 32 bytes
};

} // namespace lowered
} // namespace snippets
} // namespace ov
