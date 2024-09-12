// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/pass/pass.hpp"

namespace ov {
namespace intel_cpu {
namespace pass {

/**
 * @interface InsertBrgemmCopyBBuffers
 * @brief Insert Buffers after BrgemmCopyB with algorithm of allocation size calculation which
 *        distinguishes with common algorithm
 * @ingroup snippets
 */
class InsertBrgemmCopyBBuffers: public snippets::lowered::pass::RangedPass {
public:
    InsertBrgemmCopyBBuffers() = default;
    OPENVINO_RTTI("InsertBrgemmCopyBBuffers", "Pass");
    bool run(snippets::lowered::LinearIR& linear_ir, snippets::lowered::LinearIR::constExprIt begin, snippets::lowered::LinearIR::constExprIt end) override;
};

}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov
