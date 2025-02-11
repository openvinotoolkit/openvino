// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/itt.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/lowered/specific_loop_iter_handlers.hpp"
#include "snippets/lowered/pass/iter_handler.hpp"
#include "snippets/op/brgemm.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface BrgemmBlockingBase
 * @brief Base class for Brgemm blocking, which defines interface for blocking markup,
 *        and contains default implementation
 * @ingroup snippets
 */
class BrgemmBlockingBase {
public:
    static snippets::lowered::SpecificIterationHandlers get_default_blocking_loop_handlers(size_t work_amount, size_t block_size);

protected:
    /**
     * @interface get_blocking_params
     * @brief Computes optimal blocking params for current brgemm expression
     * @param brgemm_expr Brgemm expression
     * @return tuple in format (m_block, n_block, k_block)
     */
    virtual std::tuple<size_t, size_t, size_t> get_blocking_params(const ov::snippets::lowered::ExpressionPtr& brgemm_expr) const;
    /**
     * @interface get_brgemm_dimensions
     * @brief Extract current dimensions M,N,K of `brgemm_expr`
     * @param brgemm_expr Brgemm expression
     * @return tuple in format (M, N, K)
     */
    static std::tuple<size_t, size_t, size_t> get_brgemm_dimensions(const ov::snippets::lowered::ExpressionPtr& brgemm_expr);
    /**
     * @interface mark_blocking_loops
     * @brief Covers brgemm with blocking loops. Also should calculate optimal blocking parameters inside.
     * @param linear_ir LIR that contains brgemm
     * @param brgemm_it iterator on brgemm expression which should be covered with blocking loops
     */
    virtual bool mark_blocking_loops(snippets::lowered::LinearIR& linear_ir,
                                     const snippets::lowered::LinearIR::constExprIt& brgemm_it,
                                     size_t m_block,
                                     size_t n_block,
                                     size_t k_block);

    static bool blocking_loop_exists(const snippets::lowered::LoopManagerPtr& loop_manager,
                                     const ov::snippets::lowered::ExpressionPtr& brgemm_expr);

    void mark_m_blocking(const snippets::lowered::LoopManagerPtr& loop_manager,
                         snippets::lowered::LinearIR::constExprIt loop_begin,
                         snippets::lowered::LinearIR::constExprIt loop_end,
                         const std::vector<snippets::lowered::LoopPort>& entries,
                         const std::vector<snippets::lowered::LoopPort>& exits,
                         size_t block_size_m);

    void mark_n_blocking(const snippets::lowered::LoopManagerPtr& loop_manager,
                         snippets::lowered::LinearIR::constExprIt loop_begin,
                         snippets::lowered::LinearIR::constExprIt loop_end,
                         const std::vector<snippets::lowered::LoopPort>& entries,
                         const std::vector<snippets::lowered::LoopPort>& exits,
                         size_t block_size_n);

    void mark_k_blocking(const snippets::lowered::LoopManagerPtr& loop_manager,
                         snippets::lowered::LinearIR::constExprIt loop_begin,
                         snippets::lowered::LinearIR::constExprIt loop_end,
                         const std::vector<snippets::lowered::LoopPort>& entries,
                         const std::vector<snippets::lowered::LoopPort>& exits,
                         size_t block_size_k);

    virtual SpecificIterationHandlers get_m_loop_handlers(size_t work_amount, size_t block_size) const;
    virtual SpecificIterationHandlers get_n_loop_handlers(size_t work_amount, size_t block_size) const;
    virtual SpecificIterationHandlers get_k_loop_handlers(size_t work_amount, size_t block_size) const;

    virtual size_t get_default_m_blk(size_t m) const;
    virtual size_t get_default_n_blk(size_t n) const;
    virtual size_t get_default_k_blk(size_t k) const;
};

/**
 * @interface BrgemmBlocking
 * @brief Base class for brgemm blocking passes
 * @ingroup snippets
 */
template <typename BRGEMM_TYPE,
          typename std::enable_if<std::is_base_of<ov::snippets::op::Brgemm, BRGEMM_TYPE>::value, bool>::type = true>
class BrgemmBlocking : public snippets::lowered::pass::RangedPass, public BrgemmBlockingBase {
public:
    OPENVINO_RTTI("BrgemmBlocking", "", RangedPass)

    bool run(snippets::lowered::LinearIR& linear_ir,
             snippets::lowered::LinearIR::constExprIt begin,
             snippets::lowered::LinearIR::constExprIt end) override final {  // NOLINT
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::BrgemmBlocking")
        const auto& loop_manager = linear_ir.get_loop_manager();
        bool modified = false;
        for (auto expr_it = begin; expr_it != end; expr_it++) {
            const auto& brgemm_expr = *expr_it;
            const auto brgemm = ov::as_type_ptr<BRGEMM_TYPE>(brgemm_expr->get_node());
            if (!brgemm)
                continue;
            OPENVINO_ASSERT(!blocking_loop_exists(loop_manager, brgemm_expr),
                            "Brgemm mustn't be covered in loops before blocking pass");
            size_t m_block, n_block, k_block;
            std::tie(m_block, n_block, k_block) = get_blocking_params(brgemm_expr);
            modified = mark_blocking_loops(linear_ir, expr_it, m_block, n_block, k_block);
        }
        return modified;
    }
};
} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov