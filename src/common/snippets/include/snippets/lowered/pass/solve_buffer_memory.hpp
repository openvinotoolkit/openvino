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
 * @brief The pass optimally calculates the common buffer scratchpad size and
 *        set the offsets relative to the common data pointer to all Buffers. The pass uses MemorySolver API.
 *        Note: The pass requires expression enumeration. It should be executed seperately before this pass!
 * @ingroup snippets
 */
class SolveBufferMemory : public Pass {
public:
    OPENVINO_RTTI("SolveBufferMemory", "Pass")
    SolveBufferMemory(const AllocateBufferMemory::BufferClusters& clusters) : m_clusters(clusters) {}
    /**
     * @brief Apply the pass to the Linear IR
     * @param linear_ir the target Linear IR
     * @return status of the pass
     */
    bool run(lowered::LinearIR& linear_ir) override;
    /**
     * @brief Get buffer scratchpad size in bytes
     * @return the size of buffer sratchpad in bytes
     */
    size_t get_scratchpad_size() const { return m_buffer_scratchpad_size; }

private:
    /**
     * @brief Initializes boxes for MemorySolver
     * @param buffer_clusters buffer clusters. Can be goten using DefineBufferClusters pass
     * @return vector of boxes for MemorySolver
     */
    std::vector<MemorySolver::Box> init_boxes(const AllocateBufferMemory::BufferClusters& buffer_clusters);

    size_t m_buffer_scratchpad_size = 0;
    AllocateBufferMemory::BufferClusters m_clusters;
    constexpr static size_t m_alignment = 32; // 32 bytes
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
