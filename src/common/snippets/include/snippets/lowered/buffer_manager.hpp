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
    BufferManager() = default;

    static int64_t allocate(const lowered::LinearIR& linear_ir);

private:
    using BufferCluster = std::set<ExpressionPtr>;
    using BufferClusters = std::vector<BufferCluster>;

    static size_t init_default(const lowered::LinearIR& linear_ir);
    static BufferClusters init_clusters(const lowered::LinearIR& linear_ir);
    static std::vector<MemorySolver::Box> init_boxes(const BufferClusters& buffer_clusters);
    static void set_buffer_offset(const ExpressionPtr& buffer_expr, const size_t offset);

    constexpr static bool m_enable_optimizations = true;
    constexpr static int64_t m_alignment = 32; // 32 bytes
};

} // namespace lowered
} // namespace snippets
} // namespace ov
