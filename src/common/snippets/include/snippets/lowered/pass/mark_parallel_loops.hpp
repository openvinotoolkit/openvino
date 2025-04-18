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
 * @interface MarkParallelLoops
 * @brief Converts outermost loops to parallel to enable parallel execution
 * @ingroup snippets
 */
class MarkParallelLoops : public RangedPass {
public:
    OPENVINO_RTTI("MarkParallelLoops", "", RangedPass);
    MarkParallelLoops() = default;
    bool run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) override;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
