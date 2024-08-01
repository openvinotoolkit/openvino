// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/pass/pass.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {
/**
 * @interface UpdateMemoryAccessCounts
 * @brief The pass changes counts of all MemoryAccess ops
 * @attention The pass skips inner loops
 * @attention The pass ignores memory access ports which have count == 1
 * @param m_count - count which must be set
 * @ingroup snippets
 */
class UpdateMemoryAccessCounts : public pass::RangedPass {
public:
    UpdateMemoryAccessCounts(size_t count);
    OPENVINO_RTTI("UpdateMemoryAccessCounts", "RangedPass")
    bool run(LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) override;
    std::shared_ptr<pass::PassBase> merge(const std::shared_ptr<pass::PassBase>& other) override;

private:
    size_t m_count;
};

/**
 * @interface SetFillOffset
 * @brief The pass changes offset of all Fill ops
 * @param m_offset - offset which must be set
 * @ingroup snippets
 */
class SetFillOffset : public pass::RangedPass {
public:
    SetFillOffset(size_t offset);
    OPENVINO_RTTI("SetFillOffset", "RangedPass")
    bool run(LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) override;
    std::shared_ptr<pass::PassBase> merge(const std::shared_ptr<pass::PassBase>& other) override;

private:
    size_t m_offset;
};

/**
 * @interface TransformInnerSplitLoop
 * @brief The pass updates finalization offsets, work amount and increment of inner Loop basing on tail_size of the current Loop
 * @param m_tail_size - tail_size of the current Loop
 * @ingroup snippets
 */
class TransformInnerSplitLoop : public pass::RangedPass {
public:
    TransformInnerSplitLoop(size_t tail_size);
    OPENVINO_RTTI("TransformInnerSplitLoop", "RangedPass")
    bool run(LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) override;
    std::shared_ptr<pass::PassBase> merge(const std::shared_ptr<pass::PassBase>& other) override;

private:
    size_t m_tail_size;
};

/**
 * @interface SetEvaluateOnce
 * @brief The pass set `evaluate once = true` only to ExpandedLoopInfo which is mapped on LoopEnd in the passed iterator `end`.
 *        The pointer arithmetic should be updated in the separate optimization `OptimizeLoopSingleEvaluation`
 * @ingroup snippets
 */
class SetEvaluateOnce : public snippets::lowered::pass::RangedPass {
public:
    SetEvaluateOnce() = default;
    OPENVINO_RTTI("SetEvaluateOnce", "RangedPass")
    bool run(snippets::lowered::LinearIR& linear_ir,
             snippets::lowered::LinearIR::constExprIt begin,
             snippets::lowered::LinearIR::constExprIt end) override;
    std::shared_ptr<snippets::lowered::pass::PassBase> merge(const std::shared_ptr<snippets::lowered::pass::PassBase>& other) override;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov