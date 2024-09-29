// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"
#include "snippets/lowered/loop_manager.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface SplitLoops
 * @brief If Unified Loop `loop_1` has larger increment but the same works amount of Unified loop `loop_2`, that follows loop_1, then split loop_2
 *        into two loops so the outermost of the split loops could be fused with the loop_1 using the pass `FuseLoops`.
 * Example:
 *         Loop_1_begin                                Loop_1_begin                                  Loop_1_begin
 *              ...                                        ...                                           ...
 *         Loop_1_end (wa = 128, inc = 32)             Loop_1_end (wa = 128, inc = 32)               Split_loop_2_begin
 *              ...                         Splitting      ...                               Fusing      ...
 *         Loop_2_begin                         =>     Split_loop_1_begin                      =>    Split_loop_2_end (wa = 32, inc = 1)
 *              ...                                    Split_loop_2_begin                                ...
 *         Loop_2_end (wa = 128, inc = 1)                  ...                                       Loop_1_end (wa = 128, inc = 32)
 *                                                     Split_loop_2_end (wa = 32, inc = 1)
 *                                                     Split_loop_1_end (wa = 128, inc = 32)
 * @ingroup snippets
 */

class SplitLoops : public RangedPass {
public:
    OPENVINO_RTTI("SplitLoops", "RangedPass")
    SplitLoops() = default;
    bool run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) override;

private:
    static bool can_be_split(const UnifiedLoopInfoPtr& current, const UnifiedLoopInfoPtr& target);

    static void split(LinearIR& linear_ir, size_t loop_to_split_id, size_t outer_increment);

    /**
     * @interface TransformInnerSplitLoop
     * @brief The pass replace existing inner splitted LoopInfo with new InnerSplittedUnifiedLoopInfo and
     *        update the corresponding LoopInfo
     * @ingroup snippets
     */
    class TransformInnerSplitLoop : public pass::RangedPass {
    public:
        TransformInnerSplitLoop() = default;
        OPENVINO_RTTI("TransformInnerSplitLoop", "RangedPass")
        bool run(LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) override;
        std::shared_ptr<pass::PassBase> merge(const std::shared_ptr<pass::PassBase>& other) override;
    };
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov