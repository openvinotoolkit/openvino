// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"
#include "snippets/snippets_isa.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface AllocateBufferMemory
 * @brief Allocates memory for Buffer ops using the following optimizations:
 *         - MemorySolver: helps to solve issue of optimal memory allocation;
 *         - InPlace: Loop or MemoryAccess ops read from the memory and store data to the same memory if possible
 *         - Reusing Buffer IDs: Buffers have the same IDs (gpr) in cases when Buffers aren't connected or has the same data ptr shifts
 *        Note: All buffers are related to each other and represent common buffer scratchpad of Subgraph.
 *              The buffer scratchpad has one general data pointer. Each buffer has offset relative to the data pointer of buffer scratchpad.
 * @ingroup snippets
 */
class AllocateBufferMemory : public Pass {
public:
    OPENVINO_RTTI("AllocateBufferMemory", "Pass")
    AllocateBufferMemory(size_t& buffer_scratchpad_size, bool is_optimized = true)
        : m_buffer_scratchpad_size(buffer_scratchpad_size), m_is_optimized(is_optimized) {
        m_buffer_scratchpad_size = 0;
    }
    /**
     * @brief Apply the pass to the Linear IR
     * @param linear_ir the target Linear IR
     * @return status of the pass
     */
    bool run(lowered::LinearIR& linear_ir) override;

    /**
     * @brief Set offset to Buffer op and propagates its to the connected memory access ops
     * @param buffer_expr expression with Buffer op
     * @param offset offset in common buffer scratchpad
     */
    static void set_buffer_offset(const ExpressionPtr& buffer_expr, const size_t offset);

    using BufferCluster = std::set<ExpressionPtr>;
    using BufferClusters = std::vector<BufferCluster>;

private:
    // [113536]: DefineBufferClusters and SolveBufferMemory passes require that exressions in LinearIR have execution order.
    //           Temporary solution is resuing of topological order from tokenization.
    class EnumerateExprs : public Pass {
        OPENVINO_RTTI("EnumerateExprs", "Pass")
        bool run(lowered::LinearIR& linear_ir) override;
    };

    size_t& m_buffer_scratchpad_size;
    // Debug parameter
    //  - If True (default value), the pass activates all possible optimizations described above
    //  - If False, the pass sets uniqie ID, offsets to each Buffer in Linear IR
    bool m_is_optimized = true;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
