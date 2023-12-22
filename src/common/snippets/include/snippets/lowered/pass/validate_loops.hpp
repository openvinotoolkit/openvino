// Copyright (C) 2023 Intel Corporation
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
 * @interface ValidateLoops
 * @brief The pass validates LoopInfo that describes Loops:
 *          - Verifies the correctness of nested Loops.
 *            The loops with the same dimension index (splitted dimension) should be successively nested
 *          - dim_idx are sorted in accordance with loop nesting
 *        Notes:
 *          - should be called before explicit Loop insertion in LinearIR (InitLoop pass call).
 *          - TODO [112196] : probably, it's a temporary design. Need to investigate it and remove these limitations
 * @ingroup snippets
 */
class ValidateLoops : public Pass {
public:
    OPENVINO_RTTI("ValidateLoops", "Pass")
    ValidateLoops();
    bool run(LinearIR& linear_ir) override;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
