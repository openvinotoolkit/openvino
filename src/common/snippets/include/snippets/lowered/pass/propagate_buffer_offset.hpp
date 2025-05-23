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
class PropagateBufferOffset: public Pass {
public:
    OPENVINO_RTTI("PropagateBufferOffset", "", Pass);
    PropagateBufferOffset() = default;

    /**
     * @brief Apply the pass to the Linear IR
     * @param linear_ir the target Linear IR
     * @return status of the pass
     */
    bool run(LinearIR& linear_ir) override;

private:
    /**
     * @brief Propagates Buffer offset to the connected memory access ops
     * @param buffer_expr expression with Buffer op with inited offset
     */
    static void propagate(const BufferExpressionPtr& buffer_expr);
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
