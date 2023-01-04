// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "linear_IR_transformation.hpp"
#include "snippets/snippets_isa.hpp"

namespace ngraph {
namespace snippets {
namespace pass {
namespace lowered {

/**
 * @interface PropagateOffsetAndResetBuffer
 * @brief Propagates Buffer offsets to connected Load/Store (and other MemoryAccess) operations.
 *        Also, calculates the amount of data stored to the Buffer (via Store inside one or more Loops),
 *        and resets the corresponding pointer (sets negative finalization offset to the outermost LoopEnd).
 * @ingroup snippets
 */

class PropagateOffsetAndResetBuffer : public LinearIRTransformation {
    static void propagate_offset(const LoweredExprIR& linear_ir, const LoweredExprPtr& buffer_expr, size_t offset);
    size_t m_buffer_scratchpad_size = 0;

public:
    OPENVINO_RTTI("PropagateOffsetAndResetBuffer", "LinearIRTransformation")
    bool run(LoweredExprIR& linear_ir) override;
    size_t get_scratchpad_size() const {return m_buffer_scratchpad_size;}
};

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph
