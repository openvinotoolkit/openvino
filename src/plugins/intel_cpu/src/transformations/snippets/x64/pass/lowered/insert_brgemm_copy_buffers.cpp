// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "insert_brgemm_copy_buffers.hpp"

#include "snippets/lowered/loop_manager.hpp"
#include "snippets/itt.hpp"

#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"
#include "expressions/brgemm_copy_b_buffer_expressions.hpp"


using namespace ov::intel_cpu::brgemm_utils::repacking;
using namespace ov::snippets::lowered;

namespace ov {
namespace intel_cpu {
namespace pass {

bool InsertBrgemmCopyBuffers::run(LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::InsertBrgemmCopyBuffers")

    const auto& factory = linear_ir.get_expr_factory();

    auto insert_buffer = [&](const ExpressionPtr& copy_b_expr, size_t out_port, LinearIR::constExprIt insertion_pos) {
        const auto& copy_b = ov::as_type_ptr<ov::intel_cpu::BrgemmCopyB>(copy_b_expr->get_node());
        const auto& copy_b_out = copy_b_expr->get_output_port_connector(out_port);
        const auto copy_b_consumers = copy_b_out->get_consumers();
        OPENVINO_ASSERT(copy_b_consumers.size() == 1, "BufferCopyB must have only one consumer on each out port - Brgemm");
        const auto& buffer_op = std::make_shared<ov::snippets::op::Buffer>(copy_b->output(out_port));
        BufferExpressionPtr buffer_expr = nullptr;
        if (out_port == 0) {
            buffer_expr = factory->build<RepackedWeightsBufferExpression>(buffer_op, {copy_b_out});
        } else if (out_port == 1 && with_compensations(copy_b->get_type())) {
            buffer_expr = factory->build<CompensationsBufferExpression>(buffer_op, {copy_b_out});
        } else {
            OPENVINO_THROW("BrgemmCopyB has incorrect output ports");
        }
        return linear_ir.insert_expr(buffer_expr, LoopManager::get_common_outer_loops(copy_b_expr, copy_b_consumers.begin()->get_expr()),
                                     true, insertion_pos, {copy_b_consumers});
    };

    auto update_scratchpad = [](const ExpressionPtr& brgemm_expr, const BufferExpressionPtr& scratch_expr) {
        OPENVINO_ASSERT(scratch_expr && scratch_expr->is_independent_memory(), "Incorrect Scratchpad buffer for Brgemm AMX");
        const auto src_dt = brgemm_expr->get_node()->get_input_element_type(0);
        const auto in_subtensor = ov::snippets::utils::get_projected_subtensor(brgemm_expr->get_input_port(0));
        const auto shape0 = ov::snippets::utils::get_planar_vdims(brgemm_expr->get_input_port(0));
        const auto K_dim = shape0.back();
        const auto M_blk = *++in_subtensor.rbegin();
        OPENVINO_ASSERT(!ov::snippets::utils::is_dynamic_value(M_blk), "M blk cannot be dynamic!");

        const auto inner_k_blk = brgemm_utils::repacking::compute_inner_k_block(src_dt);
        const size_t tile_scratch_size = BrgemmCPU::SCRATCH_BYTE_SIZE;
        const size_t repacked_in0_size = ov::snippets::utils::is_dynamic_value(K_dim) || K_dim % inner_k_blk ?
                                         (M_blk * inner_k_blk * src_dt.size()) : 0;
        scratch_expr->set_allocation_size(tile_scratch_size + repacked_in0_size);
    };

    bool modified = false;
    for (auto expr_it = begin; expr_it != end; ++expr_it) {
        const auto expr = *expr_it;
        if (const auto brgemm_cpu = ov::as_type_ptr<ov::intel_cpu::BrgemmCPU>(expr->get_node())) {
            if (brgemm_utils::with_amx(brgemm_cpu->get_type())) {
                const auto& scratch_expr =
                    ov::as_type_ptr<ov::snippets::lowered::BufferExpression>(expr->get_input_port_connector(2)->get_source().get_expr());
                update_scratchpad(expr, scratch_expr);
                modified = true;
            }
        }

        if (auto copy_b = ov::as_type_ptr<ov::intel_cpu::BrgemmCopyB>(expr->get_node())) {
            for (size_t i = 0; i < expr->get_output_count(); ++i) {
                expr_it = insert_buffer(expr, i, std::next(expr_it));
            }
            modified = true;
        }
    }
    return modified;
}

}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov

