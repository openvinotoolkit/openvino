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
 * @interface BufferAllocation
 * @brief The pass calculation common size of buffer scratchpad and propagates Buffer offsets to connected MemoryAccess operations.
 * @ingroup snippets
 */

class BufferAllocation : public LinearIRTransformation {
    static void propagate_offset(const LoweredExprIR& linear_ir, const LoweredExprPtr& buffer_expr, size_t offset);
    size_t m_buffer_scratchpad_size = 0;

public:
    OPENVINO_RTTI("BufferAllocation", "LinearIRTransformation")
    bool run(LoweredExprIR& linear_ir) override;
    size_t get_scratchpad_size() const {return m_buffer_scratchpad_size;}
};

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph
