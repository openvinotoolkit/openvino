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
 * @interface InitBuffers
 * @brief The pass inits Buffer scratchpad default: Buffers have different IDs (registers) and offsets.
 * @ingroup snippets
 */

class InitBuffers : public Pass {
public:
    OPENVINO_RTTI("InitBuffers", "Pass")
    /**
     * @brief Apply the pass to the Linear IR
     * @param linear_ir the target Linear IR
     * @return status of the pass
     */
    bool run(lowered::LinearIR& linear_ir) override;

    /**
     * @brief Get buffer scratchpad size in bytes
     * @return the size of buffer sratchpad in bytes
     */
    size_t get_scratchpad_size() const { return m_buffer_scratchpad_size; }

private:
    size_t m_buffer_scratchpad_size = 0;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
