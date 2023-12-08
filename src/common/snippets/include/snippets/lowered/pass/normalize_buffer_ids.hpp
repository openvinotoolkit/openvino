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
 * @interface NormalizeBufferIDs
 * @brief After optimizations some Buffer IDs might be set unevenly: some numbers are missed.
 *        For example,
 *                 [Buffer -> ID]
 *                  Buffer0 -> 0    Two Buffers have ID = 0, one has ID = 2.
 *                  Buffer1 -> 2    Obviosly, we can normalize this IDs to set ID = 1 to Buffer1.
 *                  Buffer2 -> 0    It helps to assign GPR registers in `AssignRegister` more effective.
 *        Thus, the pass normalize IDs of Buffers in Linear IR.
 * @ingroup snippets
 */

class NormalizeBufferIDs : public Pass {
public:
    OPENVINO_RTTI("NormalizeBufferIDs", "Pass")
    /**
     * @brief Apply the pass to the Linear IR
     * @param linear_ir the target Linear IR
     * @return status of the pass
     */
    bool run(lowered::LinearIR& linear_ir) override;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
