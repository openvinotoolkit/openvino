// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/loop_info.hpp"
#include "snippets/lowered/pass/pass.hpp"

namespace ov {
namespace intel_cpu {
namespace pass {

/**
 * @interface BuildBrgemm
 * @brief The pass explicitly insert LoadBegin and LoadEnd in Linear IR using UnifiedLoopInfo from Loop markup algorithm
 * @ingroup snippets
 */
class BuildBrgemm : public snippets::lowered::pass::RangedPass {
public:
    OPENVINO_RTTI("BuildBrgemm", "", snippets::lowered::pass::RangedPass)
    BuildBrgemm() = default;
    bool run(snippets::lowered::LinearIR& linear_ir,
             snippets::lowered::LinearIR::constExprIt begin,
             snippets::lowered::LinearIR::constExprIt end) override;

private:
    std::unordered_set<snippets::lowered::UnifiedLoopInfoPtr> m_affected_loops;
};

} // namespace pass
} // namespace intel_cpu
} // namespace ov
