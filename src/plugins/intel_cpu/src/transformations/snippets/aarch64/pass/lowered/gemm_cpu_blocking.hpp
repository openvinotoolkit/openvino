// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/pass/brgemm_blocking.hpp"
#include "snippets/lowered/specific_loop_iter_handlers.hpp"
#include "transformations/snippets/aarch64/op/gemm_cpu.hpp"

namespace ov::intel_cpu::pass {

/**
 * @interface GemmCPUBlocking
 * @brief Covers GemmCPU with blocking loops
 * @ingroup snippets
 */
class GemmCPUBlocking : public ov::snippets::lowered::pass::BrgemmBlocking<ov::intel_cpu::aarch64::GemmCPU> {
public:
    OPENVINO_RTTI("GemmCPUBlocking", "", BrgemmBlocking)

    /**
     * @interface DummyPass
     * @brief The empty pass which is used to force insertion of first specific iteration of loop by K dimension
     *        Note: the pass is in public section to have opportunity to validate blocking loop in tests
     * @ingroup snippets
     */
    class DummyPass : public snippets::lowered::pass::RangedPass {
    public:
        DummyPass() = default;
        OPENVINO_RTTI("DummyPass", "", snippets::lowered::pass::RangedPass)
        bool run(snippets::lowered::LinearIR& linear_ir,
                 snippets::lowered::LinearIR::constExprIt begin,
                 snippets::lowered::LinearIR::constExprIt end) override;
        std::shared_ptr<snippets::lowered::pass::PassBase> merge(
            const std::shared_ptr<snippets::lowered::pass::PassBase>& other) override;
    };

private:
    snippets::lowered::SpecificIterationHandlers get_k_loop_handlers(size_t work_amount,
                                                                     size_t block_size) const override;

    std::tuple<size_t, size_t, size_t> get_blocking_params(
        const ov::snippets::lowered::ExpressionPtr& gemm_expr) const override;
};

}  // namespace ov::intel_cpu::pass
