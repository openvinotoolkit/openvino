// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/pass/pass.hpp"
#include "snippets/lowered/loop_info.hpp"

namespace ov {
namespace intel_cpu {
namespace pass {

/**
 * @interface AdjustBrgemmCopyBLoopPorts
 * @brief BrgemmCopyB is located outside of blocking loops and repacks input data into a blocked layout.
 *        This layout should be accounted for when we increment BrgemmCopyB data pointers. This pass
 *        Finds loop ports connected to BrgemmCopyB and sets appropriate pointer increments.
  * @ingroup snippets
 */
class AdjustBrgemmCopyBLoopPorts: public snippets::lowered::pass::Pass {
public:
    AdjustBrgemmCopyBLoopPorts() = default;
    OPENVINO_RTTI("AdjustBrgemmCopyBLoopPorts", "RangedPass");
    bool run(const snippets::lowered::LinearIR& linear_ir);
    bool run(snippets::lowered::LinearIR& linear_ir) override {
        return run(const_cast<const snippets::lowered::LinearIR&>(linear_ir));
    }
    static bool update_loop_info(const std::shared_ptr<snippets::lowered::UnifiedLoopInfo>& uni_loop_info);
    const std::unordered_set<std::shared_ptr<snippets::lowered::UnifiedLoopInfo>>& get_affected_loops() { return m_affected_loops; }
private:
    std::unordered_set<std::shared_ptr<snippets::lowered::UnifiedLoopInfo>> m_affected_loops;
};

}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov
