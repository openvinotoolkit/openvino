// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "insert_brgemm_copy_buffers.hpp"

#include "expressions/brgemm_copy_b_buffer_expressions.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"

using namespace ov::intel_cpu::brgemm_utils::repacking;
using namespace ov::snippets::lowered;

namespace ov::intel_cpu::pass {

bool InsertBrgemmCopyBuffers::run(LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::InsertBrgemmCopyBuffers")

    const auto& factory = linear_ir.get_expr_factory();

    auto insert_copy_b_buffer =
        [&](const ExpressionPtr& copy_b_expr, size_t out_port, LinearIR::constExprIt insertion_pos) {
            const auto& copy_b = ov::as_type_ptr<ov::intel_cpu::BrgemmCopyB>(copy_b_expr->get_node());
            const auto& copy_b_out = copy_b_expr->get_output_port_connector(out_port);
            const auto copy_b_consumers = copy_b_out->get_consumers();
            OPENVINO_ASSERT(copy_b_consumers.size() == 1,
                            "BufferCopyB must have only one consumer on each out port - Brgemm");
            const auto& buffer_op = std::make_shared<ov::snippets::op::Buffer>(copy_b->output(out_port));
            BufferExpressionPtr buffer_expr = nullptr;
            if (out_port == 0) {
                buffer_expr = factory->build<RepackedWeightsBufferExpression>(buffer_op, {copy_b_out});
            } else if (out_port == 1 && with_compensations(copy_b->get_type())) {
                buffer_expr = factory->build<CompensationsBufferExpression>(buffer_op, {copy_b_out});
            } else {
                OPENVINO_THROW("BrgemmCopyB has incorrect output ports");
            }
            return linear_ir.insert_expr(
                buffer_expr,
                LoopManager::get_common_outer_loops(copy_b_expr, copy_b_consumers.begin()->get_expr()),
                true,
                insertion_pos,
                {copy_b_consumers});
        };

    auto update_scratchpad = [](const ExpressionPtr& brgemm_expr, const BufferExpressionPtr& scratch_expr) {
        OPENVINO_ASSERT(scratch_expr && scratch_expr->is_independent_memory(),
                        "Incorrect Scratchpad buffer for Brgemm AMX");
        const auto src_dt = brgemm_expr->get_node()->get_input_element_type(0);
        const auto in_subtensor = ov::snippets::utils::get_projected_subtensor(brgemm_expr->get_input_port(0));
        const auto shape0 = ov::snippets::utils::get_planar_vdims(brgemm_expr->get_input_port(0));
        const auto K_dim = shape0.back();
        const auto M_blk = *++in_subtensor.rbegin();
        OPENVINO_ASSERT(!ov::snippets::utils::is_dynamic_value(M_blk), "M blk cannot be dynamic!");

        const auto vnni_factor = brgemm_utils::compute_vnni_factor(src_dt);
        const auto inner_k_blk = brgemm_utils::repacking::compute_inner_k_block(src_dt);
        OPENVINO_ASSERT(inner_k_blk > 0, "inner_k_blk cannot be zero!");
        const auto tile_scratch_size = BrgemmCPU::SCRATCH_BYTE_SIZE;
        const auto current_scratch_size = scratch_expr->get_byte_size();
        OPENVINO_ASSERT(current_scratch_size == tile_scratch_size,
                        "Tile scratchpad for BrgemmAMX should have byte size ",
                        tile_scratch_size);
        size_t inner_k_size = 0;
        if (ov::snippets::utils::is_dynamic_value(K_dim)) {
            // In dynamic case we don't know exactly if we need repacking of MatMul first input.
            // Because of that, we allocate maximum possible size for repacked data in compilation stage.
            inner_k_size = inner_k_blk;
        } else {
            // In static case, we allocate buffer for repacked data only if we have to repack MatMul first input:
            // only if `K_dim % inner_k_blk > 0`
            const auto inner_k_tail = K_dim % inner_k_blk;
            inner_k_size = inner_k_tail % vnni_factor > 0 ? ov::snippets::utils::rnd_up(inner_k_tail, vnni_factor) : 0;
        }
        const auto repacked_in0_size = M_blk * inner_k_size * src_dt.size();
        scratch_expr->set_allocation_size(tile_scratch_size + repacked_in0_size);
    };

    bool modified = false;
    for (auto expr_it = begin; expr_it != end; ++expr_it) {
        const auto& brgemm_expr = *expr_it;
        if (const auto brgemm_cpu = ov::as_type_ptr<ov::intel_cpu::BrgemmCPU>(brgemm_expr->get_node())) {
            if (brgemm_utils::with_repacking(brgemm_cpu->get_type())) {
                // BrgemmCopyB might be extracted from the body
                if (const auto copy_b_expr = brgemm_utils::repacking::get_copy_b_expr(brgemm_expr)) {
                    auto insertion_it = std::next(linear_ir.find_before(expr_it, copy_b_expr));
                    for (size_t i = 0; i < copy_b_expr->get_output_count(); ++i) {
                        insertion_it = std::next(insert_copy_b_buffer(copy_b_expr, i, insertion_it));
                    }
                    modified = true;
                }
            }

            if (brgemm_utils::with_amx(brgemm_cpu->get_type())) {
                const auto& scratch_expr = ov::as_type_ptr<ov::snippets::lowered::BufferExpression>(
                    brgemm_expr->get_input_port_connector(2)->get_source().get_expr());
                update_scratchpad(brgemm_expr, scratch_expr);
                modified = true;
            }
        }
    }
    return modified;
}

}  // namespace ov::intel_cpu::pass
