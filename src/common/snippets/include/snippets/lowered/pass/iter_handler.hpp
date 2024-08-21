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
 * @interface SetLoopIncrementOne
 * @brief The pass set `increment = 1` to ExpandedLoopInfo which is mapped on LoopEnd in the passed iterator `end` and to this LoopEnd.
 * @ingroup snippets
 */
class SetLoopIncrementOne : public snippets::lowered::pass::RangedPass {
public:
    SetLoopIncrementOne() = default;
    OPENVINO_RTTI("SetLoopIncrementOne", "RangedPass")
    bool run(snippets::lowered::LinearIR& linear_ir,
             snippets::lowered::LinearIR::constExprIt begin,
             snippets::lowered::LinearIR::constExprIt end) override;
    std::shared_ptr<snippets::lowered::pass::PassBase> merge(const std::shared_ptr<snippets::lowered::pass::PassBase>& other) override;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov