// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/rtti.hpp"
#include "pass.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"

namespace ov::snippets::lowered::pass {

/**
 * @interface ValidateUnifiedLoops
 * @brief The pass validates UnifiedLoopInfo that describes entire Loops:
 *          - Verifies the correctness of nested Loops.
 *            The loops with the same dimension index (splitted dimension) should be successively nested
 *          - dim_idx are sorted in accordance with loop nesting
 *        Notes:
 *          - should be called before explicit Loop insertion in LinearIR (InitLoop pass call).
 *          - TODO [112196] : probably, it's a temporary design. Need to investigate it and remove these limitations
 * @ingroup snippets
 */
class ValidateUnifiedLoops : public Pass {
public:
    OPENVINO_RTTI("ValidateUnifiedLoops", "", Pass)
    ValidateUnifiedLoops() = default;
    bool run(LinearIR& linear_ir) override;

private:
    static void validate_loop_infos(const LoopManagerPtr& loop_manager);
    static void validate_loop_port_presence(const LinearIR& linear_ir);
};

}  // namespace ov::snippets::lowered::pass
