// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/rtti.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/pass/pass.hpp"

namespace ov::intel_cpu::pass {

/**
 * @interface ParallelizeGatedMlpNLoops
 * @brief Enables internal parallelization for N-dimension blocking loops in GatedMLP patterns.
 * This pass identifies GatedMLP computational patterns characterized by:
 * - Exactly 3 BrgemmCPU operations
 * - First two Brgemm operations sharing the same A input
 *
 * @note The blocking loop order is assumed to be: M -> N -> K.
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

}  // namespace ov::intel_cpu::pass
