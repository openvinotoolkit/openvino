// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface AllocateBuffers
 * @brief The pass allocates common memory for all Buffers.
 *        There are two modes: default and optimized allocation. Default allocation (non-optimized) mode sets unique offsets and ID to Buffers.
 *        Optimized mode allocates memory for Buffer ops using the following optimizations:
 *         - MemorySolver: helps to solve issue of optimal memory allocation;
 *         - InPlace: Loop or MemoryAccess ops read from the memory and store data to the same memory if possible
 *         - Reusing Buffer RegGroups: Buffers have the same RegGroup (gpr) in cases when Buffers aren't connected or have the same data ptr shifts
 *        Note: All buffers are related to each other and represent common buffer scratchpad of Subgraph.
 *              The buffer scratchpad has one general data pointer. Each buffer has offset relative to the data pointer of buffer scratchpad.
 * @ingroup snippets
 */
class AllocateBuffers: public RangedPass {
public:
    OPENVINO_RTTI("AllocateBuffers", "RangedPass")
    AllocateBuffers(bool is_optimized = true);

    /**
     * @brief Apply the pass to the Linear IR
     * @param linear_ir the target Linear IR
     * @return status of the pass
     */
    bool run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) override;

private:
    bool m_is_optimized_mode = true;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
