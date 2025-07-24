// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/brgemm_blocking.hpp"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <tuple>
#include <vector>

#include "openvino/core/except.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_info.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/lowered/loop_port.hpp"
#include "snippets/lowered/pass/propagate_subtensors.hpp"
#include "snippets/lowered/specific_loop_iter_handlers.hpp"
#include "snippets/lowered/specific_loop_iter_types.hpp"
#include "snippets/utils/utils.hpp"

namespace ov::snippets::lowered::pass {
using namespace ov::snippets::utils;
using PortType = LoopPort::Type;

lowered::SpecificIterationHandlers BrgemmBlockingBase::get_default_blocking_loop_handlers(size_t work_amount,
                                                                                          size_t block_size) {
    OPENVINO_ASSERT(block_size != 0, "block size must be non zero");
    SpecificIterationHandlers handlers;
    const auto tail_size =
        utils::is_dynamic_value(work_amount) ? utils::get_dynamic_value<size_t>() : work_amount % block_size;
    if (tail_size != 0) {
        handlers.register_pass<lowered::SpecificLoopIterType::LAST_ITER, lowered::pass::UpdateSubtensors>(tail_size);
    }
    return handlers;
}

bool BrgemmBlockingBase::blocking_loop_exists(const lowered::LoopManagerPtr& loop_manager,
                                              const ExpressionPtr& brgemm_expr) {
    auto check_port = [&](const LoopPort& p) {
        return p.get_expr_port()->get_expr() == brgemm_expr && any_of(p.get_dim_idx(), 0ul, 1ul);
    };
    const auto& loop_ids = brgemm_expr->get_loop_ids();
    return std::any_of(loop_ids.begin(), loop_ids.end(), [&](const auto& id) {
        const auto loop = loop_manager->get_loop_info(id);
        return std::any_of(loop->get_input_ports().begin(), loop->get_input_ports().end(), check_port) ||
               std::any_of(loop->get_output_ports().begin(), loop->get_output_ports().end(), check_port);
    });
}

void BrgemmBlockingBase::mark_m_blocking(const LoopManagerPtr& loop_manager,
                                         LinearIR::constExprIt loop_begin,
                                         LinearIR::constExprIt loop_end,
                                         const std::vector<LoopPort>& entries,
                                         const std::vector<LoopPort>& exits,
                                         size_t block_size_m) {
    const auto planar_dims = get_planar_vdims(*entries[0].get_expr_port());
    const auto m = *++planar_dims.rbegin();
    const auto id = loop_manager->mark_loop(loop_begin, loop_end, m, block_size_m, entries, exits, false);
    loop_manager->get_loop_info<UnifiedLoopInfo>(id)->set_handlers(get_m_loop_handlers(m, block_size_m));
}

void BrgemmBlockingBase::mark_n_blocking(const LoopManagerPtr& loop_manager,
                                         LinearIR::constExprIt loop_begin,
                                         LinearIR::constExprIt loop_end,
                                         const std::vector<LoopPort>& entries,
                                         const std::vector<LoopPort>& exits,
                                         size_t block_size_n) {
    const auto planar_dims = get_planar_vdims(*entries[1].get_expr_port());
    const auto n = *planar_dims.rbegin();
    const auto id = loop_manager->mark_loop(loop_begin, loop_end, n, block_size_n, entries, exits, false);
    loop_manager->get_loop_info<UnifiedLoopInfo>(id)->set_handlers(get_n_loop_handlers(n, block_size_n));
}

void BrgemmBlockingBase::mark_k_blocking(const LoopManagerPtr& loop_manager,
                                         LinearIR::constExprIt loop_begin,
                                         LinearIR::constExprIt loop_end,
                                         const std::vector<LoopPort>& entries,
                                         const std::vector<LoopPort>& exits,
                                         size_t block_size_k) {
    const auto planar_dims = get_planar_vdims(*entries[0].get_expr_port());
    const auto k = *planar_dims.rbegin();
    const auto id = loop_manager->mark_loop(loop_begin, loop_end, k, block_size_k, entries, exits, false);
    loop_manager->get_loop_info<UnifiedLoopInfo>(id)->set_handlers(get_k_loop_handlers(k, block_size_k));
}

SpecificIterationHandlers BrgemmBlockingBase::get_m_loop_handlers(size_t work_amount, size_t block_size) const {
    return get_default_blocking_loop_handlers(work_amount, block_size);
}
SpecificIterationHandlers BrgemmBlockingBase::get_n_loop_handlers(size_t work_amount, size_t block_size) const {
    return get_default_blocking_loop_handlers(work_amount, block_size);
}
SpecificIterationHandlers BrgemmBlockingBase::get_k_loop_handlers(size_t work_amount, size_t block_size) const {
    return get_default_blocking_loop_handlers(work_amount, block_size);
}

std::tuple<size_t, size_t, size_t> BrgemmBlockingBase::get_blocking_params(const ExpressionPtr& brgemm_expr) const {
    const auto [m, n, k] = get_brgemm_dimensions(brgemm_expr);

    const auto m_blk = 32;
    const auto n_blk = 64;
    const auto k_blk = !ov::snippets::utils::is_dynamic_value(k) && k > 1024 ? 1024 : 512;

    // Ticket: 113745
    // TODO: extend block size selection heuristics
    return std::make_tuple(get_corrected_blk_size_by_dim(m, m_blk),
                           get_corrected_blk_size_by_dim(n, n_blk),
                           get_corrected_blk_size_by_dim(k, k_blk));
}

std::tuple<size_t, size_t, size_t> BrgemmBlockingBase::get_brgemm_dimensions(const ExpressionPtr& brgemm_expr) {
    OPENVINO_ASSERT(brgemm_expr, "Brgemm expression is nullptr!");
    const auto& in_0_desc = brgemm_expr->get_input_port_descriptor(0);
    const auto& in_1_desc = brgemm_expr->get_input_port_descriptor(1);
    const auto& out_desc = brgemm_expr->get_output_port_descriptor(0);

    const auto& in_0_planar_dims = get_planar_vdims(in_0_desc->get_shape(), in_0_desc->get_layout());
    const auto& in_1_planar_dims = get_planar_vdims(in_1_desc->get_shape(), in_1_desc->get_layout());
    const auto& out_preordered_dims = get_preordered_vdims(out_desc->get_shape(), out_desc->get_layout());

    const auto& m = *++out_preordered_dims.rbegin();
    const auto& n = *out_preordered_dims.rbegin();
    const auto& k0 = *in_0_planar_dims.rbegin();
    const auto& k1 = *++in_1_planar_dims.rbegin();
    size_t k = 0;
    OPENVINO_ASSERT(utils::merge_dynamic_dim(k, k0, k1),
                    "Brgemm input descriptors have incompatible K dimension value.");
    return std::make_tuple(m, n, k);
}

bool BrgemmBlockingBase::mark_blocking_loops(LinearIR& linear_ir,
                                             const LinearIR::constExprIt& brgemm_it,
                                             size_t m_block,
                                             size_t n_block,
                                             size_t k_block) {
    const auto& brgemm_expr = *brgemm_it;
    brgemm_expr->get_input_port_descriptor(0)->set_subtensor(VectorDims{m_block, k_block});
    brgemm_expr->get_input_port_descriptor(1)->set_subtensor(VectorDims{k_block, n_block});
    brgemm_expr->get_output_port_descriptor(0)->set_subtensor(VectorDims{m_block, n_block});

    const auto& loop_manager = linear_ir.get_loop_manager();
    if (!is_full_dim_value(k_block)) {
        const std::vector<LoopPort> entries{LoopPort::create<PortType::Incremented>(brgemm_expr->get_input_port(0), 0),
                                            LoopPort::create<PortType::Incremented>(brgemm_expr->get_input_port(1), 1)};
        const std::vector<LoopPort> exits{LoopPort::create<PortType::NotProcessed>(brgemm_expr->get_output_port(0))};
        mark_k_blocking(loop_manager, brgemm_it, std::next(brgemm_it), entries, exits, k_block);
    }
    if (!is_full_dim_value(n_block)) {
        const std::vector<LoopPort> entries{LoopPort::create<PortType::NotProcessed>(brgemm_expr->get_input_port(0)),
                                            LoopPort::create<PortType::Incremented>(brgemm_expr->get_input_port(1))};
        const std::vector<LoopPort> exits{LoopPort::create<PortType::Incremented>(brgemm_expr->get_output_port(0))};
        mark_n_blocking(loop_manager, brgemm_it, std::next(brgemm_it), entries, exits, n_block);
    }
    if (!is_full_dim_value(m_block)) {
        const std::vector<LoopPort> entries{LoopPort::create<PortType::Incremented>(brgemm_expr->get_input_port(0), 1),
                                            LoopPort::create<PortType::NotProcessed>(brgemm_expr->get_input_port(1))};
        const std::vector<LoopPort> exits{LoopPort::create<PortType::Incremented>(brgemm_expr->get_output_port(0), 1)};
        mark_m_blocking(loop_manager, brgemm_it, std::next(brgemm_it), entries, exits, m_block);
    }
    return true;
}

size_t BrgemmBlockingBase::get_corrected_blk_size_by_dim(size_t dim, size_t default_blk) {
    if (!utils::is_dynamic_value(dim) && dim <= default_blk) {
        return utils::get_full_dim_value();
    }
    return default_blk;
}

}  // namespace ov::snippets::lowered::pass
