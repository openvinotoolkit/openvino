// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/pass/pass.hpp"

namespace ov::intel_cpu::pass {

/**
 * @interface InsertBrgemmCopyBuffers
 * @brief Insert Brgemm-specific buffers:
 *          - after BrgemmCopyB with algorithm of allocation size calculation which distinguishes with common algorithm
 *          - update size of `NewMemory` Buffer - add allocation byte size for repacked data from first input of Brgemm
 * in AMX scenario
 * @ingroup snippets
 */
class InsertBrgemmCopyBuffers : public snippets::lowered::pass::RangedPass {
public:
    InsertBrgemmCopyBuffers() = default;
    OPENVINO_RTTI("InsertBrgemmCopyBuffers", "", RangedPass)
    bool run(snippets::lowered::LinearIR& linear_ir,
             snippets::lowered::LinearIR::constExprIt begin,
             snippets::lowered::LinearIR::constExprIt end) override;
};

}  // namespace ov::intel_cpu::pass
