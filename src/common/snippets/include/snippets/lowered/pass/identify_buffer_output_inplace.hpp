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
 * @interface IdentifyBufferOutputInplace
 * @brief This pass identify if buffer could share subgraph output memory.
 *        Set the m_inplace_flag to the output number that could share memory to buffer.
 * @ingroup snippets
 */
class IdentifyBufferOutputInplace : public Pass {
public:
    OPENVINO_RTTI("IdentifyBufferOutputInplace", "Pass")

    IdentifyBufferOutputInplace(int& inplace_flag) : m_inplace_flag(inplace_flag) {}
    /**
     * @brief Apply the pass to the Linear IR
     * @param linear_ir the target Linear IR
     * @return status of the pass
     */
    bool run(lowered::LinearIR& linear_ir) override;

private:
    int& m_inplace_flag;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
