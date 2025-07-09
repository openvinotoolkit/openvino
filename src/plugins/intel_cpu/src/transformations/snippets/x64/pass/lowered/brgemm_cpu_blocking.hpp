// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <tuple>

#include "openvino/core/rtti.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/pass/brgemm_blocking.hpp"
#include "snippets/lowered/pass/pass.hpp"
#include "snippets/lowered/specific_loop_iter_handlers.hpp"
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

    static bool is_kn_blocking_supported(const ov::element::Type& input_type);

private:
    static snippets::lowered::LinearIR::constExprIt move_new_memory_buffer(
        snippets::lowered::LinearIR& linear_ir,
        const snippets::lowered::LinearIR::constExprIt& brgemm_it);

    /**
     * @brief Updates the loop information for the specified loop IDs with new ports.
     *
     * @param loop_manager The loop manager
     * @param loop_ids A vector of loop IDs whose information needs to be updated.
     * @param block_to_new_ports A vector of pairs where each pair consists of:
     *        - block size (e.g., m_block, n_block, or k_block).
     *        - vector of new LoopPort objects to be added to the loop.
     */
    static void update_loop_infos(
        const ov::snippets::lowered::LoopManagerPtr& loop_manager,
        const std::vector<size_t>& loop_ids,
        const std::vector<std::pair<size_t, std::vector<ov::snippets::lowered::LoopPort>>>& block_to_new_ports);

    /**
     * @brief Create new ports for not processed postops.
     * @note Postop ports are supported by blocking pass only as not processed
     */
    static void create_not_processed_postops_ports(const snippets::lowered::ExpressionPtr& brgemm_expr,
                                                   const snippets::lowered::LoopManagerPtr& loop_manager,
                                                   size_t m_block,
                                                   size_t n_block,
                                                   size_t k_block);

    snippets::lowered::SpecificIterationHandlers get_k_loop_handlers(size_t work_amount,
                                                                     size_t block_size) const override;

    std::tuple<size_t, size_t, size_t> get_blocking_params(
        const ov::snippets::lowered::ExpressionPtr& brgemm_expr) const override;
    bool mark_blocking_loops(snippets::lowered::LinearIR& linear_ir,
                             const snippets::lowered::LinearIR::constExprIt& brgemm_it,
                             size_t m_block,
                             size_t n_block,
                             size_t k_block) override;
};

}  // namespace ov::intel_cpu::pass
