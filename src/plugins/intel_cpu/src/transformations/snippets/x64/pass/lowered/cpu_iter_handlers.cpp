// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_iter_handlers.hpp"

#include "snippets/op/loop.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"

namespace ov {
namespace intel_cpu {
namespace pass {
using LinearIR = snippets::lowered::LinearIR;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;

SetBrgemmBeta::SetBrgemmBeta(float beta) : snippets::lowered::pass::RangedPass(), m_beta(beta) {}

bool SetBrgemmBeta::run(LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) {
    for (auto expr_it = begin; expr_it != end; ++expr_it) {
        const auto& expr = expr_it->get();
        if (const auto brgemm = ov::as_type_ptr<ov::snippets::op::Brgemm>(expr->get_node())) {
            brgemm->set_beta(m_beta);
        }
    }
    return true;
}

std::shared_ptr<snippets::lowered::pass::PassBase> SetBrgemmBeta::merge(const std::shared_ptr<snippets::lowered::pass::PassBase>& other) {
    const auto merged_pass = std::make_shared<SetBrgemmBeta>(m_beta);
    if (other == nullptr)
        return merged_pass;
    const auto casted_pass = ov::as_type_ptr<SetBrgemmBeta>(other);
    if (!casted_pass || m_beta != casted_pass->m_beta)
        return nullptr;
    return merged_pass;
}

SetEvaluanceOnce::SetEvaluanceOnce(bool evaluation) : snippets::lowered::pass::RangedPass(), m_evaluation(evaluation) {}

bool SetEvaluanceOnce::run(LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) {
    const auto& loop_end = ov::as_type_ptr<snippets::op::LoopEnd>(end->get()->get_node());
    OPENVINO_ASSERT(loop_end, "SetEvaluanceOnce expected LoopEnd node in iterator `end`.");
    const auto& loop_info = linear_ir.get_loop_manager()->get_loop_info<ov::snippets::lowered::ExpandedLoopInfo>(loop_end->get_id());
    loop_info->set_evaluate_once(m_evaluation);
    return true;
}

std::shared_ptr<snippets::lowered::pass::PassBase> SetEvaluanceOnce::merge(const std::shared_ptr<snippets::lowered::pass::PassBase>& other) {
    const auto merged_pass = std::make_shared<SetEvaluanceOnce>(m_evaluation);
    if (other == nullptr)
        return merged_pass;
    const auto casted_pass = ov::as_type_ptr<SetEvaluanceOnce>(other);
    if (!casted_pass || m_evaluation != casted_pass->m_evaluation)
        return nullptr;
    return merged_pass;
}

}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov