// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "insert_brgemm_copy_b_buffers.hpp"

#include "snippets/lowered/loop_manager.hpp"
#include "snippets/itt.hpp"

#include "transformations/snippets/x64/op/brgemm_copy_a.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"
#include "expressions/brgemm_copy_buffer_expressions.hpp"


using namespace ov::intel_cpu::brgemm_utils::repacking;
using namespace ov::snippets::lowered;

namespace ov {
namespace intel_cpu {
namespace pass {

bool InsertBrgemmCopyBBuffers::run(LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::InsertBrgemmCopyBBuffers")

    const auto& factory = linear_ir.get_expr_factory();

    auto insert_buffer = [&](const ExpressionPtr& expr, size_t out_port, LinearIR::constExprIt insertion_pos) {
        const auto& out = expr->get_output_port_connector(out_port);
        const auto consumers = out->get_consumers();
        OPENVINO_ASSERT(consumers.size() == 1, "BrgemmCopyA and BrgemmCopyB must have only one consumer on each out port - Buffer");
        const auto& buffer_op = std::make_shared<ov::snippets::op::Buffer>(expr->get_node()->output(out_port));
        BufferExpressionPtr buffer_expr = nullptr;
        if (ov::is_type<ov::intel_cpu::BrgemmCopyA>(expr->get_node())) {
            OPENVINO_ASSERT(out_port == 0, "BrgemmCopyA must have only one output!");
            buffer_expr = factory->build<RepackedActivationsBufferExpression>(buffer_op, {out});
        } else if (const auto brgemm_copy_b_expr = ov::as_type_ptr<ov::intel_cpu::BrgemmCopyB>(expr->get_node())) {
            if (out_port == 0) {
                buffer_expr = factory->build<RepackedWeightsBufferExpression>(buffer_op, {out});
            } else if (out_port == 1 && brgemm_copy_b_expr->get_config().need_compensations()) {
                buffer_expr = factory->build<CompensationsBufferExpression>(buffer_op, {out});
            } else {
                OPENVINO_THROW("BrgemmCopyB has incorrect output ports");
            }
        }
        return linear_ir.insert_expr(buffer_expr, LoopManager::get_common_outer_loops(expr, consumers.begin()->get_expr()), true, insertion_pos, {consumers});
    };

    bool modified = false;
    for (auto expr_it = begin; expr_it != end; ++expr_it) {
        const auto expr = *expr_it;
        if (ov::is_type<ov::intel_cpu::BrgemmCopyA>(expr->get_node()) || ov::is_type<ov::intel_cpu::BrgemmCopyB>(expr->get_node())) {
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

