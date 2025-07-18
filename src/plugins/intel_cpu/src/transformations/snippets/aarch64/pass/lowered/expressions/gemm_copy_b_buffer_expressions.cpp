// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gemm_copy_b_buffer_expressions.hpp"

#include <cstddef>
#include <memory>

#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/expressions/buffer_expression.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/shape_inference/shape_inference.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/aarch64/op/gemm_copy_b.hpp"
#include "transformations/snippets/aarch64/op/gemm_cpu.hpp"
#include "utils/general_utils.h"

using namespace ov::snippets::lowered;

namespace ov::intel_cpu::aarch64 {

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
    OPENVINO_ASSERT(ov::is_type<ov::intel_cpu::aarch64::GemmCopyB>(parent_out.get_expr()->get_node()) &&
                        parent_out.get_index() == 0,
                    "RepackedWeightsBufferExpression expects GemmCopyB as parent expression");
    OPENVINO_ASSERT(any_of(get_node()->get_input_element_type(0), ov::element::f32),
                    "RepackedWeightsBufferExpression after GemmCopyB currently only support f32 data type on arm");
}

void RepackedWeightsBufferExpression::init_allocation_size(
    [[maybe_unused]] const std::shared_ptr<snippets::lowered::LoopManager>& loop_manager,
    [[maybe_unused]] size_t allocation_rank) {
    const auto& parent_expr = get_input_port_connector(0)->get_source().get_expr();
    const auto& in_shape = ov::snippets::utils::get_planar_vdims(parent_expr->get_input_port(0));
    OPENVINO_ASSERT(in_shape.size() >= 2 && allocation_rank >= 2, "GemmCopyB should has at least 2 rank tensor");
    const auto& element_type = get_node()->get_input_element_type(0);
    const size_t N = *in_shape.rbegin();
    const size_t K = *++in_shape.rbegin();

    const auto& consumers = get_output_port_connector(0)->get_consumers();
    ExpressionPtr child_gemm_expr = nullptr;
    // maybe connected to loopEnd besides gemm
    for (const auto& consumer : consumers) {
        if (ov::is_type<ov::intel_cpu::aarch64::GemmCPU>(consumer.get_expr()->get_node())) {
            child_gemm_expr = consumer.get_expr();
            break;
        }
    }
    OPENVINO_ASSERT(child_gemm_expr, "RepackedWeightsBufferExpression must connect to gemm");
    const auto& gemm_in_subtensor = ov::snippets::utils::get_projected_subtensor(child_gemm_expr->get_input_port(1));
    const size_t n_block_size = *gemm_in_subtensor.rbegin();
    if (snippets::utils::is_dynamic_value(N) || snippets::utils::is_dynamic_value(K) ||
        snippets::utils::is_dynamic_value(n_block_size)) {
        m_allocation_size = snippets::utils::get_dynamic_value<size_t>();
        return;
    }
    size_t n_block_num = N / n_block_size;
    size_t n_tail_size = N % n_block_size;
    m_allocation_size = n_block_num * kai_get_rhs_packed_size_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(n_block_size, K);
    if (n_tail_size > 0) {
        m_allocation_size += kai_get_rhs_packed_size_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(n_tail_size, K);
    }
    // convert byte size to element type size
    m_allocation_size = m_allocation_size / element_type.size();
}

}  // namespace ov::intel_cpu::aarch64
