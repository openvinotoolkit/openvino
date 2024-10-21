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
 * @brief
 * 1. If Unified Loop `loop_1` has larger increment but the same works amount of Unified loop `loop_2`, that follows loop_1, then split loop_2
 *        into two loops so the outermost of the split loops could be fused with the loop_1 using the pass `FuseLoops`.
 * Example such as enable Q block fusion in MHA:
 *         Loop_1_begin                                Loop_1_begin                                  Loop_1_begin
 *              ...                                        ...                                           ...
 *         Loop_1_end (wa = 128, inc = 32)             Loop_1_end (wa = 128, inc = 32)               Split_loop_2_begin
 *              ...                         Splitting      ...                               Fusing      ...
 *         Loop_2_begin                         =>     Split_loop_1_begin                      =>    Split_loop_2_end (wa = 32, inc = 1)
 *              ...                                    Split_loop_2_begin                                ...
 *         Loop_2_end (wa = 128, inc = 1)                  ...                                       Loop_1_end (wa = 128, inc = 32)
 *                                                     Split_loop_2_end (wa = 32, inc = 1)
 *                                                     Split_loop_1_end (wa = 128, inc = 32)
 * Example such as enable K/V block fusion in MHA:
 *         Loop_1_begin                                Loop_1_begin                                  Loop_1_begin
 *              ...                                        ...                                           ...
 *         Loop_1_end (wa = 1024, inc = 256)           Loop_1_end (wa = 1024, inc = 256)             Split_loop_2_begin
 *              ...                         Splitting      ...                               Fusing      ...
 *         Loop_2_begin                         =>     Split_loop_1_begin                      =>    Split_loop_2_end (wa = 256, inc = 16)
 *              ...                                    Split_loop_2_begin                                ...
 *         Loop_2_end (wa = 1024, inc = 16)                ...                                       Loop_1_end (wa = 1024, inc = 256)
 *                                                     Split_loop_2_end (wa = 256, inc = 16)
 *                                                     Split_loop_1_end (wa = 1024, inc = 256)
 * MM0 is M and N split, MM1 is M and K split
 * split ... + softmax on M amd N/K dimension
 *             N,N_blk  M,M_blk              N,vec  M,1                  K,K_blk  M,M_blk
 * |__|__MM0___|________|      |__|___Elt____|______|      |_|__MM1______|________|
 *
 *             N,N_blk  M,M_blk               N,vec M_blk,1 M,M_blk                K,K_blk  M,M_blk
 * |__|__MM0___|________|      |_|_|___Elt____|_____|_______|        |_|___MM1_____|________|
 *
 *             N,N_blk  M,M_blk                  N_blk,vec N,N_blk   M_blk,1 M,M_blk              K,K_blk  M,M_blk
 * |__|__MM0___|________|      |_|_|_|____Elt____|_________|_________|_______|      |_|__MM1______|________|
 *
 *             N,N_blk  M,M_blk                  N_blk,vec M_blk,1   N,N_blk M,M_blk              K,K_blk  M,M_blk
 * |__|__MM0___|________|      |_|_|_|___Elt_____|_________|_________|_______|      |_|__MM1______|________|
 *
 *                        N_blk,vec  M_blk,1             N,N_blk  M,M_blk   (K and N is the same)
 * |__|__MM0___|_|__Elt___|__________|________MM1________|________|
 * @ingroup snippets
 */

class SplitLoops : public RangedPass {
public:
    OPENVINO_RTTI("SplitLoops", "RangedPass")
    SplitLoops() = default;
    bool run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) override;

private:
    static bool can_be_split(const UnifiedLoopInfoPtr& current, const UnifiedLoopInfoPtr& target);

    static size_t split(LinearIR& linear_ir, size_t loop_to_split_id, size_t outer_increment, size_t loop_position);

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