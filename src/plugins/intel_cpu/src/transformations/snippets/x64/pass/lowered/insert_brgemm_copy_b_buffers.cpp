// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "insert_brgemm_copy_b_buffers.hpp"

#include "snippets/lowered/loop_manager.hpp"
#include "snippets/itt.hpp"

#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"
#include "expressions/brgemm_copy_b_buffer_expressions.hpp"


using namespace ov::intel_cpu::brgemm_utils::repacking;
using namespace ov::snippets::lowered;

namespace ov {
namespace intel_cpu {
namespace pass {

bool InsertBrgemmCopyBBuffers::run(LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::InsertBrgemmCopyBBuffers")

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

    bool modified = false;
    for (auto expr_it = begin; expr_it != end; ++expr_it) {
        const auto expr = *expr_it;
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

