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
 * @ingroup snippets
 */

/**
 * MM0 is M and N split, MM1 is M and K split
 * split ... + softmax + ... on second dimension
*/

class SplitLoops : public RangedPass {
public:
    OPENVINO_RTTI("SplitLoops", "RangedPass")
    SplitLoops();
    bool run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) override;

private:
    static bool can_be_split(const UnifiedLoopInfoPtr& current, const UnifiedLoopInfoPtr& target);
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov