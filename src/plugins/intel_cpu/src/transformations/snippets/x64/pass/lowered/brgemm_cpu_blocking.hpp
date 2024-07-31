// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/pass/brgemm_blocking.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"

namespace ov {
namespace intel_cpu {
namespace pass {

/**
 * @interface BrgemmCPUBlocking
 * @brief Covers BrgemmCPU with blocking loops
 * @ingroup snippets
 */
class BrgemmCPUBlocking : public ov::snippets::lowered::pass::BrgemmBlocking<BrgemmCPU> {
public:
    OPENVINO_RTTI("BrgemmCPUBlocking", "BrgemmBlocking")

private:
    static snippets::lowered::LinearIR::constExprIt move_new_memory_buffer(snippets::lowered::LinearIR& linear_ir,
                                                                           const snippets::lowered::LinearIR::constExprIt& brgemm_it);

    static snippets::lowered::LinearIR::constExprIt get_loop_begin_pos(snippets::lowered::LinearIR& linear_ir,
                                                                       const snippets::lowered::LinearIR::constExprIt& brgemm_it,
                                                                       const snippets::lowered::ExpressionPtr& copy_b_expr);

    std::tuple<size_t, size_t, size_t> get_blocking_params(const ov::snippets::lowered::ExpressionPtr& brgemm_expr) override;
    bool mark_blocking_loops(snippets::lowered::LinearIR& linear_ir,
                             const snippets::lowered::LinearIR::constExprIt& brgemm_it,
                             size_t m_block,
                             size_t n_block,
                             size_t k_block) override;
};

}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov