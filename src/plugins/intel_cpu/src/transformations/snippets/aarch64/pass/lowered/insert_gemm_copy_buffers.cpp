// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "insert_gemm_copy_buffers.hpp"

#include <iterator>
#include <memory>

#include "expressions/gemm_copy_b_buffer_expressions.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/expressions/buffer_expression.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/op/buffer.hpp"
#include "transformations/snippets/aarch64/op/gemm_copy_b.hpp"
#include "transformations/snippets/aarch64/op/gemm_cpu.hpp"
#include "transformations/snippets/aarch64/op/gemm_utils.hpp"

using namespace ov::intel_cpu::aarch64::gemm_utils::repacking;
using namespace ov::snippets::lowered;

namespace ov::intel_cpu::pass::aarch64 {

bool InsertGemmCopyBuffers::run(LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::InsertBrgemmCopyBuffers")

    const auto& factory = linear_ir.get_expr_factory();

    auto insert_copy_b_buffer = [&](const ExpressionPtr& copy_b_expr, LinearIR::constExprIt insertion_pos) {
        OPENVINO_ASSERT(copy_b_expr->get_output_count() == 1, "gemm copyb must have only one output");
        const auto& copy_b = ov::as_type_ptr<ov::intel_cpu::aarch64::GemmCopyB>(copy_b_expr->get_node());
        const auto& copy_b_out = copy_b_expr->get_output_port_connector(0);
        const auto copy_b_consumers = copy_b_out->get_consumers();
        OPENVINO_ASSERT(copy_b_consumers.size() == 1,
                        "BufferCopyB must have only one consumer on each out port - Gemm");
        const auto& buffer_op = std::make_shared<ov::snippets::op::Buffer>(copy_b->output(0));
        BufferExpressionPtr buffer_expr =
            factory->build<ov::intel_cpu::aarch64::RepackedWeightsBufferExpression>(buffer_op, {copy_b_out});
        linear_ir.insert_expr(buffer_expr,
                              LoopManager::get_common_outer_loops(copy_b_expr, copy_b_consumers.begin()->get_expr()),
                              true,
                              insertion_pos,
                              {copy_b_consumers});
    };

    auto insert_nullified_bias_buffer = [&](const ExpressionPtr& copy_b_expr) {
        OPENVINO_ASSERT(copy_b_expr->get_input_count() == 2, "GemmCopyB should have two inputs");
        const auto& current_buffer_expr = copy_b_expr->get_input_port_connector(1)->get_source().get_expr();
        OPENVINO_ASSERT(ov::is_type<ov::snippets::op::Buffer>(current_buffer_expr->get_node()),
                        "Expected Buffer on bias input");
        const auto& empty_buffer_op = std::make_shared<ov::snippets::op::Buffer>(ov::Shape{1});
        BufferExpressionPtr new_buffer_expr =
            factory->build<ov::intel_cpu::aarch64::NullifiedBiasBufferExpression>(empty_buffer_op, {});
        new_buffer_expr->set_loop_ids(current_buffer_expr->get_loop_ids());
        linear_ir.replace_with_expr({current_buffer_expr}, new_buffer_expr);
        // Force allocation size recalculation
        new_buffer_expr->init_allocation_size(linear_ir.get_loop_manager(), linear_ir.get_config().m_loop_depth);
    };

    bool modified = false;
    for (auto expr_it = begin; expr_it != end; ++expr_it) {
        const auto& gemm_expr = *expr_it;
        if (const auto gemm_cpu = ov::as_type_ptr<ov::intel_cpu::aarch64::GemmCPU>(gemm_expr->get_node())) {
            if (const auto copy_b_expr = get_copy_b_expr(gemm_expr)) {
                auto insertion_it = std::next(linear_ir.find_before(expr_it, copy_b_expr));
                insert_copy_b_buffer(copy_b_expr, insertion_it);
                insert_nullified_bias_buffer(copy_b_expr);
                modified = true;
            } else {
                OPENVINO_THROW("GemmCopyB must connect to gemmCPU in subgraph, and not be extracted from the body");
            }
        }
    }
    return modified;
}

}  // namespace ov::intel_cpu::pass::aarch64
