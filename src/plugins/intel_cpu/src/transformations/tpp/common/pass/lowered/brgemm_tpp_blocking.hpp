// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/pass/brgemm_blocking.hpp"
#include "transformations/tpp/common/op/brgemm.hpp"

namespace ov::intel_cpu::tpp::pass {

/**
 * @interface BrgemmTPPBlocking
 * @brief Covers BrgemmTPP with blocking loops
 * @ingroup snippets
 */

class BrgemmTPPBlocking : public ov::snippets::lowered::pass::BrgemmBlocking<ov::intel_cpu::tpp::op::BrgemmTPP> {
public:
    OPENVINO_RTTI("BrgemmTPPBlocking",
                  "tpp::op::BrgemmTPP",
                  snippets::lowered::pass::BrgemmBlocking<ov::intel_cpu::tpp::op::BrgemmTPP>);

    /**
     * @interface SetBrgemmBeta
     * @brief The pass set `beta = 0` to BrgemmTPP.
     *        Note: the pass is in public section to have opportunity to validate blocking loop in tests
     * @ingroup snippets
     */
    class SetBrgemmBeta : public snippets::lowered::pass::RangedPass {
    public:
        OPENVINO_RTTI("SetBrgemmBeta", "0", snippets::lowered::pass::RangedPass);
        SetBrgemmBeta() = default;
        bool run(ov::snippets::lowered::LinearIR& linear_ir,
                 ov::snippets::lowered::LinearIR::constExprIt begin,
                 ov::snippets::lowered::LinearIR::constExprIt end) override;
        std::shared_ptr<snippets::lowered::pass::PassBase> merge(
            const std::shared_ptr<snippets::lowered::pass::PassBase>& other) override;
    };

private:
    [[nodiscard]] std::tuple<size_t, size_t, size_t> get_blocking_params(
        const ov::snippets::lowered::ExpressionPtr& brgemm_expr) const override;
    [[nodiscard]] ov::snippets::lowered::SpecificIterationHandlers get_k_loop_handlers(
        size_t work_amount,
        size_t block_size) const override;
};

}  // namespace ov::intel_cpu::tpp::pass
