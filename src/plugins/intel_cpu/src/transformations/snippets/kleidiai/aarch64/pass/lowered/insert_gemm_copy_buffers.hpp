// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/pass/pass.hpp"

namespace ov::intel_cpu::pass::aarch64 {

/**
 * @interface InsertGemmCopyBuffers
 * @brief Insert Gemm-specific buffers after GemmCopyB with algorithm of allocation size calculation which distinguishes
 * with common algorithm
 * @ingroup snippets
 */
class InsertGemmCopyBuffers : public snippets::lowered::pass::RangedPass {
public:
    InsertGemmCopyBuffers() = default;
    OPENVINO_RTTI("InsertGemmCopyBuffers", "", RangedPass)
    bool run(snippets::lowered::LinearIR& linear_ir,
             snippets::lowered::LinearIR::constExprIt begin,
             snippets::lowered::LinearIR::constExprIt end) override;
};

}  // namespace ov::intel_cpu::pass::aarch64
