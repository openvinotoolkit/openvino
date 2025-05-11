// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/compute_buffer_allocation_size.hpp"

#include "snippets/op/buffer.hpp"
#include "snippets/utils/utils.hpp"
#include "snippets/itt.hpp"


namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

bool ComputeBufferAllocationSize::run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::ComputeBufferAllocationSize")

    const auto& allocation_rank = linear_ir.get_config().m_loop_depth;
    const auto& loop_manager = linear_ir.get_loop_manager();
    for (const auto& buffer_expr : linear_ir.get_buffers()) {
        // If the current size is undefined, update it
        // TODO [143395] : MemoryManager will return container with only dynamic buffers without any `is_defined()`
        if (!buffer_expr->is_defined())
            buffer_expr->init_allocation_size(loop_manager, allocation_rank);
    }

    return true;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
