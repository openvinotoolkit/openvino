// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_copy_b_buffer_expressions.hpp"

#include <memory>

#include "snippets/lowered/loop_manager.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"
#include "utils/general_utils.h"

using namespace ov::intel_cpu::brgemm_utils::repacking;
using namespace ov::snippets::lowered;

namespace ov::intel_cpu {

RepackedWeightsBufferExpression::RepackedWeightsBufferExpression(
    const std::shared_ptr<ov::Node>& n,
    const std::shared_ptr<snippets::IShapeInferSnippetsFactory>& factory)
    : BufferExpression(n, factory) {}

snippets::lowered::ExpressionPtr RepackedWeightsBufferExpression::clone() const {
    return std::make_shared<RepackedWeightsBufferExpression>(*this);
}

void RepackedWeightsBufferExpression::validate() const {
    BufferExpression::validate();
    OPENVINO_ASSERT(get_input_count() == 1, "RepackedWeightsBufferExpression must have only one input");
    const auto& parent_out = get_input_port_connector(0)->get_source();
    OPENVINO_ASSERT(
        ov::is_type<ov::intel_cpu::BrgemmCopyB>(parent_out.get_expr()->get_node()) && parent_out.get_index() == 0,
        "RepackedWeightsBufferExpression expects BrgemmCopyB as parent expression");
}

void RepackedWeightsBufferExpression::init_allocation_size(
    const std::shared_ptr<snippets::lowered::LoopManager>& loop_manager,
    size_t allocation_rank) {
    const auto& parent_expr = get_input_port_connector(0)->get_source().get_expr();
    const auto& in_layout = parent_expr->get_input_port_descriptor(0)->get_layout();
    const auto& in_subtensor = ov::snippets::utils::get_projected_subtensor(parent_expr->get_input_port(0));

    const size_t n_blk = *in_subtensor.rbegin();
    const size_t k_blk = *++in_subtensor.rbegin();

    const auto& precision = get_node()->get_input_element_type(0);
    const auto buffer_b_shape =
        brgemm_utils::repacking::compute_buffer_b_allocation_shape(k_blk,
                                                                   n_blk,
                                                                   precision,
                                                                   BrgemmCopyB::is_transposed(in_layout));
    OPENVINO_ASSERT(buffer_b_shape.size() == 3, "Unexpected buffer B shape rank");
    m_allocation_size =
        std::accumulate(buffer_b_shape.cbegin(), buffer_b_shape.cend(), size_t(1), [](size_t a, size_t b) {
            return snippets::utils::dynamic_safe_mul(a, b);
        });
}

CompensationsBufferExpression::CompensationsBufferExpression(
    const std::shared_ptr<ov::Node>& n,
    const std::shared_ptr<snippets::IShapeInferSnippetsFactory>& factory)
    : BufferExpression(n, factory) {}

snippets::lowered::ExpressionPtr CompensationsBufferExpression::clone() const {
    return std::make_shared<CompensationsBufferExpression>(*this);
}

void CompensationsBufferExpression::validate() const {
    BufferExpression::validate();
    OPENVINO_ASSERT(get_input_count() == 1, "CompensationsBufferExpression must have only one input");
    const auto& parent_out = get_input_port_connector(0)->get_source();
    OPENVINO_ASSERT(
        ov::is_type<ov::intel_cpu::BrgemmCopyB>(parent_out.get_expr()->get_node()) && parent_out.get_index() == 1,
        "CompensationsBufferExpression expects BrgemmCopyB as parent expression");
}

void CompensationsBufferExpression::init_allocation_size(
    const std::shared_ptr<snippets::lowered::LoopManager>& loop_manager,
    size_t allocation_rank) {
    const auto& parent_expr = get_input_port_connector(0)->get_source().get_expr();
    // Compensations are computed during repacking, so we need to round-up allocation shape according to m_inner_n_block
    // because of OneDNN implementation nuances (as in get_repacking_buffer_size).
    // However, the compensations are computed by N dimension, so K dimension doesn't affect the compensations buffer
    const auto& precision = parent_expr->get_node()->get_input_element_type(0);
    const size_t n_blk = *ov::snippets::utils::get_projected_subtensor(parent_expr->get_input_port(0)).rbegin();
    m_allocation_size = compute_repacked_n_dim(n_blk, precision);
}

}  // namespace ov::intel_cpu
