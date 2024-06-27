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
 * @interface NormalizeBufferRegisterGroups
 * @brief After optimizations some Buffer RegGroups might be set unevenly: some numbers are missed.
 *        For example,
 *                 [Buffer -> RegGroup]
 *                  Buffer0 -> 0    Two Buffers have RegGroup = 0, one has RegGroup = 2.
 *                  Buffer1 -> 2    Obviosly, we can normalize this IDs to set RegGroup = 1 to Buffer1.
 *                  Buffer2 -> 0    It helps to assign GPR registers in `AssignRegister` more effective.
 *        Thus, the pass normalize IDs of Buffers in Linear IR.
 * @ingroup snippets
 */

class NormalizeBufferRegisterGroups : public RangedPass {
public:
    OPENVINO_RTTI("NormalizeBufferRegisterGroups", "RangedPass")
    /**
     * @brief Apply the pass to the Linear IR
     * @param linear_ir the target Linear IR
     * @return status of the pass
     */
    bool run(lowered::LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) override;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
