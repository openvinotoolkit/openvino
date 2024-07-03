// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/pass/iter_handler.hpp"

namespace ov {
namespace intel_cpu {
namespace pass {
/**
 * @interface SetBrgemmBeta
 * @brief The pass updates all CPUBrgemm nodes with a new beta value
 * @param m_beta - beta which must be set
 * @ingroup snippets
 */
class SetBrgemmBeta : public snippets::lowered::pass::RangedPass {
public:
    SetBrgemmBeta(float beta);
    OPENVINO_RTTI("SetBrgemmBeta", "RangedPass")
    bool run(snippets::lowered::LinearIR& linear_ir,
             snippets::lowered::LinearIR::constExprIt begin,
             snippets::lowered::LinearIR::constExprIt end) override;
    std::shared_ptr<snippets::lowered::pass::PassBase> merge(const std::shared_ptr<snippets::lowered::pass::PassBase>& other) override;

private:
    float m_beta = 0;
};

/**
 * @interface SetEvaluanceOnce
 * @brief The pass set `evaluate once` only to ExpandedLoopInfo which is mapped on LoopEnd in the passed iterator `end`.
 *        The pointer arithmetic should be updated in the separate optimization `OptimizeLoopSingleEvaluation`
 * @param m_evaluation - value which must be set
 * @ingroup snippets
 */
class SetEvaluanceOnce : public snippets::lowered::pass::RangedPass {
public:
    SetEvaluanceOnce(bool evaluation);
    OPENVINO_RTTI("SetEvaluanceOnce", "RangedPass")
    bool run(snippets::lowered::LinearIR& linear_ir,
             snippets::lowered::LinearIR::constExprIt begin,
             snippets::lowered::LinearIR::constExprIt end) override;
    std::shared_ptr<snippets::lowered::pass::PassBase> merge(const std::shared_ptr<snippets::lowered::pass::PassBase>& other) override;

private:
    bool m_evaluation = false;
};
}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov