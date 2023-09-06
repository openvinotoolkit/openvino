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
 * @brief TODO
 * @ingroup snippets
 */

class InitBuffers : public Pass {
public:
    OPENVINO_RTTI("InitBuffers", "Pass")
    bool run(lowered::LinearIR& linear_ir) override;

    size_t get_scratchpad_size() const { return m_buffer_scratchpad_size; }

private:
    size_t m_buffer_scratchpad_size = 0;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
