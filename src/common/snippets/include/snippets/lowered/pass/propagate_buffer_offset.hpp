// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface PropagateBufferOffset
 * @brief Get Buffer offset and set it to the ports of connected memory access operations.
 *        Should be called after `SolveMemoryBuffer`.
 * @ingroup snippets
 */
class PropagateBufferOffset: public RangedPass {
public:
    OPENVINO_RTTI("PropagateBufferOffset", "RangedPass")
    PropagateBufferOffset() = default;

    /**
     * @brief Apply the pass to the Linear IR
     * @param linear_ir the target Linear IR
     * @return status of the pass
     */
    bool run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) override;

private:
    /**
     * @brief Propagates Buffer offset to the connected memory access ops
     * @param buffer_expr expression with Buffer op with inited offset
     */
    static void propagate(const ExpressionPtr& buffer_expr);
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
