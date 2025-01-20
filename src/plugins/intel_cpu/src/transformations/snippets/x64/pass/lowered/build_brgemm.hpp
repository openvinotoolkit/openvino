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
class BuildBrgemm : public snippets::lowered::pass::ConstPass {
public:
    OPENVINO_RTTI("BuildBrgemm", "", ConstPass)
    BuildBrgemm() = default;
    bool run(const snippets::lowered::LinearIR& linear_ir) override;
    const std::unordered_set<snippets::lowered::UnifiedLoopInfoPtr>& get_affected_loops() {
        return m_affected_loops;
    }

private:
    std::unordered_set<snippets::lowered::UnifiedLoopInfoPtr> m_affected_loops;
};

} // namespace pass
} // namespace intel_cpu
} // namespace ov
