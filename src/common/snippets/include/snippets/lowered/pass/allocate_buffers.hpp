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
 * @interface AllocateBuffers
 * @brief The pass calculates common size of buffer scratchpad and propagates Buffer offsets to connected MemoryAccess operations.
 *        Notes:
 *           - The pass implicitly regulates InPlace processing for some Buffers when it's possible.
 *             The pass don't allocate new memory for InPlace Buffers, we propagate the same offsets for them.
 *           - The pass should be splitted into two passes: ProcessInplace (markup of Buffers which can use the same memory)
 *             and AllocateBuffer (allocate memory for Buffers using MemorySolver which can optimally reuse memory).
 * @ingroup snippets
 */

class AllocateBuffers : public Pass {
public:
    OPENVINO_RTTI("AllocateBuffers", "Pass")
    bool run(lowered::LinearIR& linear_ir) override;

    size_t get_scratchpad_size() const { return m_buffer_scratchpad_size; }

private:
    static void propagate_offset(const LinearIR& linear_ir, const ExpressionPtr& buffer_expr, size_t offset);

    size_t m_buffer_scratchpad_size = 0;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
