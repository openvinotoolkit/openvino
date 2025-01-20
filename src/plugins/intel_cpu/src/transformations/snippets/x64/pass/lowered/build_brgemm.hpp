// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/loop_info.hpp"
#include "snippets/lowered/pass/pass.hpp"

namespace ov::intel_cpu::pass {

/**
 * @interface BuildBrgemm
 * @brief The pass builds batch-reduce gemm (brgemm) nodes from gemm nodes.
 * @ingroup snippets
 */
class BuildBrgemm : public snippets::lowered::pass::RangedPass {
public:
    OPENVINO_RTTI("BuildBrgemm", "", snippets::lowered::pass::RangedPass)
    BuildBrgemm() = default;
    bool run(snippets::lowered::LinearIR& linear_ir,
             snippets::lowered::LinearIR::constExprIt begin,
             snippets::lowered::LinearIR::constExprIt end) override;
};

}  // namespace ov::intel_cpu::pass
