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
class AdjustBrgemmCopyBLoopPorts: public snippets::lowered::pass::RangedPass {
public:
    AdjustBrgemmCopyBLoopPorts() = default;
    OPENVINO_RTTI("AdjustBrgemmCopyBLoopPorts", "RangedPass");
    bool run(snippets::lowered::LinearIR& linear_ir,
             snippets::lowered::LinearIR::constExprIt begin,
             snippets::lowered::LinearIR::constExprIt end) override;

    using UnifiedLoopInfo = ov::snippets::lowered::UnifiedLoopInfo;
    static int get_affected_port_idx(const std::shared_ptr<UnifiedLoopInfo>& uni_loop_info);
};

}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov
