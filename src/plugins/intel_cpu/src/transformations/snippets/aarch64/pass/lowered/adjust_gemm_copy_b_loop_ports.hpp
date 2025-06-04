// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/loop_info.hpp"
#include "snippets/lowered/pass/pass.hpp"

namespace ov::intel_cpu::pass::aarch64 {

/**
 * @interface AdjustGemmCopyBLoopPorts
 * @brief GemmCopyB is located outside of blocking loops and repacks input data into a blocked layout.
 *        This layout should be accounted for when we increment GemmCopyB data pointers. This pass
 *        Finds loop ports connected to GemmCopyB and sets appropriate pointer increments.
 * @ingroup snippets
 */
class AdjustGemmCopyBLoopPorts : public snippets::lowered::pass::ConstPass {
public:
    OPENVINO_RTTI("AdjustGemmCopyBLoopPorts", "", ConstPass)
    AdjustGemmCopyBLoopPorts() = default;
    bool run(const snippets::lowered::LinearIR& linear_ir) override;
    static bool update_loop_info(const snippets::lowered::UnifiedLoopInfoPtr& uni_loop_info);
    const std::unordered_set<snippets::lowered::UnifiedLoopInfoPtr>& get_affected_loops() {
        return m_affected_loops;
    }

private:
    std::unordered_set<snippets::lowered::UnifiedLoopInfoPtr> m_affected_loops;
};

}  // namespace ov::intel_cpu::pass::aarch64
