// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"

#include "snippets/lowered/loop_manager.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface ComputeBufferAllocationSize
 * @brief The pass calculate allocation sizes of Buffers.
 * @param m_buffer_allocation_rank - rank of shape for memory allocation: shape[m_allocation_rank : -1]
 * @ingroup snippets
 */
class ComputeBufferAllocationSize : public RangedPass {
public:
    OPENVINO_RTTI("ComputeBufferAllocationSize", "", RangedPass);
    ComputeBufferAllocationSize() = default;

    bool run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) override;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
