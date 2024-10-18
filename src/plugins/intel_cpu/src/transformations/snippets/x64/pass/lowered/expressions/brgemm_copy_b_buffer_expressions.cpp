// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_copy_b_buffer_expressions.hpp"

#include "snippets/lowered/loop_manager.hpp"
#include "snippets/utils/utils.hpp"

#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"
#include "utils/general_utils.h"


using namespace ov::intel_cpu::brgemm_utils::repacking;
using namespace ov::snippets::lowered;

namespace ov {
namespace intel_cpu {

RepackedWeightsBufferExpression::RepackedWeightsBufferExpression(const std::shared_ptr<ov::Node>& n,
    const std::shared_ptr<snippets::IShapeInferSnippetsFactory>& factory) : BufferExpression(n, factory) {}

snippets::lowered::ExpressionPtr RepackedWeightsBufferExpression::clone() const {
    return std::shared_ptr<RepackedWeightsBufferExpression>(new RepackedWeightsBufferExpression(*this));
}

void RepackedWeightsBufferExpression::validate() const {
    BufferExpression::validate();
    OPENVINO_ASSERT(get_input_count() == 1, "RepackedWeightsBufferExpression must have only one input");
    const auto& parent_out = get_input_port_connector(0)->get_source();
    OPENVINO_ASSERT(ov::is_type<ov::intel_cpu::BrgemmCopyB>(parent_out.get_expr()->get_node()) && parent_out.get_index() == 0,
                    "RepackedWeightsBufferExpression expects BrgemmCopyB as parent expression");
}

void RepackedWeightsBufferExpression::init_allocation_size(const std::shared_ptr<snippets::lowered::LoopManager>& loop_manager, size_t allocation_rank) {
    const auto& parent_expr = get_input_port_connector(0)->get_source().get_expr();
    const auto& in_layout =  parent_expr->get_input_port_descriptor(0)->get_layout();
    const auto& in_subtensor = ov::snippets::utils::get_projected_subtensor(parent_expr->get_input_port(0));

    const size_t n_blk = *in_subtensor.rbegin();
    const size_t k_blk = *++in_subtensor.rbegin();

    const auto& precision = get_node()->get_input_element_type(0);
    // Repacking buffer shape is set in accordance to OneDNN requirements
    const size_t N_dim = std::max(n_blk, compute_inner_n_block(precision));
    if (!in_layout.empty() && in_layout.back() != in_layout.size() - 1) {
        // In case of transpose, K dimension must be rounded-up to number of elems in vector register
        // For the details, please see 'transpose16x8' and 'fixup16x16' implementations and usage in onednn/src/cpu/x64/matmul/brgemm_matmul_copy_utils.cpp
        const auto elems_in_vec = brgemm_utils::get_elems_in_vec(precision);
        m_allocation_size = snippets::utils::dynamic_safe_mul(N_dim, snippets::utils::rnd_up(k_blk, elems_in_vec));
    } else {
        // Low precision repacking writes the result by m_brgemmVNNIFactor * m_inner_n_block blocks
        // despite the actual size of the input data. Because of that we have to round-up the allocation shape to always have enough memory allocated.
        // For the details, please see 'copy_4x64' and 'copy_2x32' implementations and usage in onednn/src/cpu/x64/matmul/brgemm_matmul_copy_utils.cpp
        const auto brgemmVNNIFactor = brgemm_utils::compute_vnni_factor(precision);
        OPENVINO_ASSERT(brgemmVNNIFactor > 0, "brgemmVNNIFactor value must be positive.");
        m_allocation_size = snippets::utils::dynamic_safe_mul(N_dim, snippets::utils::rnd_up(k_blk, brgemmVNNIFactor));
    }
}

CompensationsBufferExpression::CompensationsBufferExpression(const std::shared_ptr<ov::Node>& n,
    const std::shared_ptr<snippets::IShapeInferSnippetsFactory>& factory) : BufferExpression(n, factory) {}

snippets::lowered::ExpressionPtr CompensationsBufferExpression::clone() const {
    return std::shared_ptr<CompensationsBufferExpression>(new CompensationsBufferExpression(*this));
}

void CompensationsBufferExpression::validate() const {
    BufferExpression::validate();
    OPENVINO_ASSERT(get_input_count() == 1, "CompensationsBufferExpression must have only one input");
    const auto& parent_out = get_input_port_connector(0)->get_source();
    OPENVINO_ASSERT(ov::is_type<ov::intel_cpu::BrgemmCopyB>(parent_out.get_expr()->get_node()) && parent_out.get_index() == 1,
                    "CompensationsBufferExpression expects BrgemmCopyB as parent expression");
}

void CompensationsBufferExpression::init_allocation_size(const std::shared_ptr<snippets::lowered::LoopManager>& loop_manager, size_t allocation_rank) {
    const auto& parent_expr = get_input_port_connector(0)->get_source().get_expr();
    // Compensations are computed during repacking, so we need to round-up allocation shape according to m_inner_n_block
    // because of OneDNN implementation nuances (as in get_repacking_buffer_size).
    // However, the compensations are computed by N dimension, so K dimension doesn't affect the compensations buffer
    const size_t n_blk = *ov::snippets::utils::get_projected_subtensor(parent_expr->get_input_port(0)).rbegin();
    if (snippets::utils::is_dynamic_value(n_blk)) {
        m_allocation_size = snippets::utils::get_dynamic_value<size_t>();
    } else {
        const auto& precision = parent_expr->get_node()->get_input_element_type(0);
        m_allocation_size = std::max(n_blk, compute_inner_n_block(precision));
    }
}

}  // namespace intel_cpu
}  // namespace ov
