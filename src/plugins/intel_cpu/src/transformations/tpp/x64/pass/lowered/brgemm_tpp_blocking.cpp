// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_tpp_blocking.hpp"

#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/tpp/x64/op/brgemm.hpp"


namespace ov {
namespace intel_cpu {
namespace tpp {
namespace pass {
using namespace ov::snippets::utils;

bool BrgemmTPPBlocking::SetBrgemmBeta::run(ov::snippets::lowered::LinearIR& linear_ir,
                                           ov::snippets::lowered::LinearIR::constExprIt begin,
                                           ov::snippets::lowered::LinearIR::constExprIt end) {
    for (auto expr_it = begin; expr_it != end; ++expr_it) {
        if (const auto brgemm = ov::as_type_ptr<ov::intel_cpu::tpp::op::BrgemmTPP>(expr_it->get()->get_node()))
            brgemm->set_beta(0);
    }
    return true;
}

std::shared_ptr<snippets::lowered::pass::PassBase> BrgemmTPPBlocking::SetBrgemmBeta::merge(const std::shared_ptr<snippets::lowered::pass::PassBase>& other) {
    return !other || ov::is_type<SetBrgemmBeta>(other) ? std::make_shared<SetBrgemmBeta>() : nullptr;
}

std::tuple<size_t, size_t, size_t> BrgemmTPPBlocking::get_blocking_params(const ov::snippets::lowered::ExpressionPtr& brgemm_expr) const {
    size_t m, n, k;
    std::tie(m, n, k) = get_brgemm_dimensions(brgemm_expr);
    OPENVINO_ASSERT(!is_dynamic_value(m) && !is_dynamic_value(n) && !is_dynamic_value(n), "BrgemmTPP doesn't support dynamic shapes");

    size_t m_blk, n_blk, k_blk;
    std::tie(m_blk, n_blk, k_blk) = BrgemmBlockingBase::get_blocking_params(brgemm_expr);

    auto get_projected_blk = [](const size_t dim, const size_t blk) { return ov::snippets::utils::is_full_dim_value(blk) ? dim : blk; };
    return std::make_tuple(get_projected_blk(m, m_blk), get_projected_blk(n, n_blk), get_projected_blk(k, k_blk));
}

ov::snippets::lowered::SpecificIterationHandlers BrgemmTPPBlocking::get_k_loop_handlers(size_t work_amount, size_t block_size) const {
    ov::snippets::lowered::SpecificIterationHandlers handlers = ov::snippets::lowered::pass::BrgemmBlockingBase::get_k_loop_handlers(work_amount, block_size);
    handlers.register_pass<ov::snippets::lowered::SpecificLoopIterType::FIRST_ITER, SetBrgemmBeta>();
    return handlers;
}
} // namespace pass
} // namespace tpp
} // namespace intel_cpu
} // namespace ov
