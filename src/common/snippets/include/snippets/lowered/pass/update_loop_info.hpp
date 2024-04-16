// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"

#include "snippets/lowered/loop_info.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface UpdateLoopInfo
 * @brief The pass update the existing UnifiedLoopInfo and the corresponding ExpandedLoopInfos.
 *        Notes:
 *          - LinearIR must have LoopManager only with ExpandedLoopInfo (LinearIR contains decomposed loops).
 *            Each of them has the pointer to UnifiedLoopInfo.
 *          - ExpandedLoopInfos` in LoopManager are sorted by execution order (NormalizeLoopIDs pas has been already passed).
 * @ingroup snippets
 */

class UpdateLoopInfo : public Pass {
public:
    OPENVINO_RTTI("UpdateLoopInfo", "Pass")
    UpdateLoopInfo() = default;
    bool run(LinearIR& linear_ir) override;

private:
    /**
     * @brief Initializes common ptr_increments and finalization offsets for ExpandedLoopInfo from ports of UnifiedLoopInfo
     * @param solid_loop_info solid LoopInfo
     * @param ptr_increments ref of vector with ptr increments
     * @param finalization_offsets ref of vector with finalization offsets
     */
    static void init_data_ptr_shifts(const UnifiedLoopInfoPtr& solid_loop_info, std::vector<int64_t>& ptr_increments,
                                     std::vector<int64_t>& finalization_offsets);
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov