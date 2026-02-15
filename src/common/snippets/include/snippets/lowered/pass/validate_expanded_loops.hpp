// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/rtti.hpp"
#include "pass.hpp"
#include "snippets/lowered/linear_ir.hpp"

namespace ov::snippets::lowered::pass {

/**
 * @interface ValidateExpandedLoops
 * @brief The pass validates loops after decomposition into specific iterations
 * @ingroup snippets
 */
class ValidateExpandedLoops : public Pass {
public:
    OPENVINO_RTTI("ValidateExpandedLoops", "", Pass)
    ValidateExpandedLoops() = default;
    bool run(LinearIR& linear_ir) override;

private:
    static void validate_loop_information(const LinearIR& linear_ir);
    static void validate_loop_expressions(const LinearIR& linear_ir);
};

}  // namespace ov::snippets::lowered::pass
