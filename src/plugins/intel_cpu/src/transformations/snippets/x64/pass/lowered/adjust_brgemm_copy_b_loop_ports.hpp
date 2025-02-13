// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/loop_info.hpp"
#include "snippets/lowered/pass/pass.hpp"

namespace ov::intel_cpu::pass {

/**
 * @interface AdjustBrgemmCopyBLoopPorts
 * @brief BrgemmCopyB is located outside of blocking loops and repacks input data into a blocked layout.
 *        This layout should be accounted for when we increment BrgemmCopyB data pointers. This pass
 *        Finds loop ports connected to BrgemmCopyB and sets appropriate pointer increments.
 * @ingroup snippets
 */
class AdjustBrgemmCopyBLoopPorts : public snippets::lowered::pass::ConstPass {
public:
    OPENVINO_RTTI("AdjustBrgemmCopyBLoopPorts", "", ConstPass)
    AdjustBrgemmCopyBLoopPorts() = default;
    bool run(const snippets::lowered::LinearIR& linear_ir) override;
    static bool update_loop_info(const snippets::lowered::UnifiedLoopInfoPtr& uni_loop_info);
    const std::unordered_set<snippets::lowered::UnifiedLoopInfoPtr>& get_affected_loops() {
        return m_affected_loops;
    }

private:
    std::unordered_set<snippets::lowered::UnifiedLoopInfoPtr> m_affected_loops;
};

}  // namespace ov::intel_cpu::pass
