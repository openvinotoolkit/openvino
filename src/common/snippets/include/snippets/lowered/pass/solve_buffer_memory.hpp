// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"

#include "allocate_buffers.hpp"
#include "openvino/runtime/memory_solver.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface SolveBufferMemory
 * @brief The pass optimally calculates the common buffer scratchpad size and
 *        set the offsets relative to the common data pointer to all Buffers. The pass uses MemorySolver API.
 *        Note: The pass requires expression enumeration. It should be executed separately before this pass!
 * @ingroup snippets
 */
class SolveBufferMemory : public Pass {
public:
    OPENVINO_RTTI("SolveBufferMemory", "Pass")

    SolveBufferMemory(size_t& buffer_scratchpad_size, AllocateBuffers::BufferClusters& clusters)
        : m_buffer_scratchpad_size(buffer_scratchpad_size), m_clusters(clusters) {}
    /**
     * @brief Apply the pass to the Linear IR
     * @param linear_ir the target Linear IR
     * @return status of the pass
     */
    bool run(lowered::LinearIR& linear_ir) override;

private:
    /**
     * @brief Initializes boxes for MemorySolver
     * @param buffer_clusters buffer clusters. These clusters could be obtained using DefineBufferClusters pass
     * @return vector of boxes for MemorySolver
     */
    std::vector<ov::MemorySolver::Box> init_boxes(const AllocateBuffers::BufferClusters& buffer_clusters);

    size_t& m_buffer_scratchpad_size;
    AllocateBuffers::BufferClusters& m_clusters;

    constexpr static size_t m_alignment = 32; // 32 bytes for data alignment in allocated memory
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
