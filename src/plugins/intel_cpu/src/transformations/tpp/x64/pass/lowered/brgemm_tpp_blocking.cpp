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

std::tuple<size_t, size_t, size_t> BrgemmTPPBlocking::get_blocking_params(const ov::snippets::lowered::ExpressionPtr& brgemm_expr) {
    const auto& in_0_desc = brgemm_expr->get_input_port_descriptor(0);
    const auto& in_1_desc = brgemm_expr->get_input_port_descriptor(1);
    const auto& out_desc = brgemm_expr->get_output_port_descriptor(0);

    const auto& in_0_planar_dims = get_planar_vdims(in_0_desc->get_shape(), in_0_desc->get_layout());
    const auto& in_1_planar_dims = get_planar_vdims(in_1_desc->get_shape(), in_1_desc->get_layout());
    const auto& out_preordered_dims = get_preordered_vdims(out_desc->get_shape(), out_desc->get_layout());

    const auto& m = *++out_preordered_dims.rbegin();
    const auto& n = *out_preordered_dims.rbegin();
    const auto& k = *in_0_planar_dims.rbegin();
    OPENVINO_ASSERT(k == *++in_1_planar_dims.rbegin(), "Brgemm input descriptors have different K dimension value.");
    OPENVINO_ASSERT(!is_dynamic_value(m) && !is_dynamic_value(n) && !is_dynamic_value(n), "BrgemmTPP doesn't support dynamic shapes");

    const auto block_size_m = std::min<size_t>(32, m);
    const auto block_size_n = std::min<size_t>(64, n);
    const auto block_size_k = k > 1024 ? 1024 : k > 512 ? 512 : k;
    return std::make_tuple(block_size_m, block_size_n, block_size_k);
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
