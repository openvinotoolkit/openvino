// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/rtti.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/pass/pass.hpp"

namespace ov::intel_cpu::pass::aarch64 {

/**
 * @interface ParallelizeGatedMlpNLoops
 * @brief Enables internal parallelization for N-dimension blocking loops in GatedMLP patterns.
 *
 * @ingroup snippets
 */
class ParallelizeGatedMlpNLoops : public snippets::lowered::pass::RangedPass {
public:
    ParallelizeGatedMlpNLoops() = default;
    OPENVINO_RTTI("ParallelizeGatedMlpNLoops", "", RangedPass)
    bool run(snippets::lowered::LinearIR& linear_ir,
             snippets::lowered::LinearIR::constExprIt begin,
             snippets::lowered::LinearIR::constExprIt end) override;
};

}  // namespace ov::intel_cpu::pass::aarch64
