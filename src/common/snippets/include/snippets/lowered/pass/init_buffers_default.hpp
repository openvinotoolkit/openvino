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
 * @interface InitBuffersDefault
 * @brief The pass inits Buffer expressions in LinearIR default (non-optimized): sets unique offsets and ID to Buffers.
 * @ingroup snippets
 */

class InitBuffersDefault : public Pass {
public:
    OPENVINO_RTTI("InitBuffersDefault", "Pass")

    InitBuffersDefault(size_t& buffer_scratchpad_size) : m_buffer_scratchpad_size(buffer_scratchpad_size) {
        m_buffer_scratchpad_size = 0;
    }
    /**
     * @brief Apply the pass to the Linear IR
     * @param linear_ir the target Linear IR
     * @return status of the pass
     */
    bool run(lowered::LinearIR& linear_ir) override;

private:
    size_t& m_buffer_scratchpad_size;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
