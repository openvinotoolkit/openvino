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
 * @brief Helps to solve issue of optimal memory allocation only for Buffers in graph using the following optimizations:
 *         - MemorySolver: helps to solve issue of optimal memory allocation;
 *         - InPlace: Loop or MemoryAccess ops read from the memory and store data to the same memory if possible
 *         - Reusing Buffer IDs: Buffers have the same IDs (gpr) in cases when Buffers aren't connected or has the same data ptr shifts
 * @ingroup snippets
 */
class AllocateBufferMemory : public Pass {
public:
    OPENVINO_RTTI("AllocateBufferMemory", "Pass")
    bool run(lowered::LinearIR& linear_ir) override;

    size_t get_scratchpad_size() const { return m_buffer_scratchpad_size; }

    static void set_buffer_offset(const ExpressionPtr& buffer_expr, const size_t offset);

    using BufferCluster = std::set<ExpressionPtr>;
    using BufferClusters = std::vector<BufferCluster>;
private:
    class EnumerateExprs : public Pass {
        OPENVINO_RTTI("EnumerateExprs", "Pass")
        bool run(lowered::LinearIR& linear_ir) override;
    };

    constexpr static bool is_optimized = true;
    size_t m_buffer_scratchpad_size = 0;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
