// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/pass/brgemm_blocking.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"

namespace ov::intel_cpu::pass {

/**
 * @interface BrgemmCPUBlocking
 * @brief Covers BrgemmCPU with blocking loops
 * @ingroup snippets
 */
class BrgemmCPUBlocking : public ov::snippets::lowered::pass::BrgemmBlocking<BrgemmCPU> {
public:
    OPENVINO_RTTI("BrgemmCPUBlocking", "", BrgemmBlocking)

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
    static snippets::lowered::LinearIR::constExprIt move_new_memory_buffer(
        snippets::lowered::LinearIR& linear_ir,
        const snippets::lowered::LinearIR::constExprIt& brgemm_it);

    snippets::lowered::SpecificIterationHandlers get_k_loop_handlers(size_t work_amount,
                                                                     size_t block_size) const override;

    std::tuple<size_t, size_t, size_t> get_blocking_params(
        const ov::snippets::lowered::ExpressionPtr& brgemm_expr) const override;
    bool mark_blocking_loops(snippets::lowered::LinearIR& linear_ir,
                             const snippets::lowered::LinearIR::constExprIt& brgemm_it,
                             size_t m_block,
                             size_t n_block,
                             size_t k_block) override;

    size_t get_default_n_blk(size_t n) const override;
};

}  // namespace ov::intel_cpu::pass
