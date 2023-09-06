// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"

#include "snippets/lowered/pass/allocate_buffer_memory.hpp"

#include "memory_solver.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface SolveBufferMemory
 * @brief TODO
 * @ingroup snippets
 */

class SolveBufferMemory : public Pass {
public:
    OPENVINO_RTTI("SolveBufferMemory", "Pass")
    SolveBufferMemory(const AllocateBufferMemory::BufferClusters& clusters) : m_clusters(clusters) {}
    bool run(lowered::LinearIR& linear_ir) override;

    size_t get_scratchpad_size() const { return m_buffer_scratchpad_size; }

private:
    std::vector<MemorySolver::Box> init_boxes(const AllocateBufferMemory::BufferClusters& buffer_clusters);

    size_t m_buffer_scratchpad_size = 0;
    AllocateBufferMemory::BufferClusters m_clusters;
    constexpr static int64_t m_alignment = 32; // 32 bytes
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
