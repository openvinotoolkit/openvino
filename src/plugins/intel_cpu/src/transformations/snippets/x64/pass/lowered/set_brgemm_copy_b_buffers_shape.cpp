// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"

#include "set_brgemm_copy_b_buffers_shape.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils.hpp"

#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"

bool ov::intel_cpu::pass::SetBrgemmCopyBBuffersShape::run(snippets::lowered::LinearIR& linear_ir,
                                                          snippets::lowered::LinearIR::constExprIt begin,
                                                          snippets::lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::SetBrgemmCopyBBuffersShape")

    auto get_buffer_from_output = [](const snippets::lowered::ExpressionPtr& expr, const size_t out_idx) {
        const auto& consumers = expr->get_output_port_connector(out_idx)->get_consumers();
        OPENVINO_ASSERT(consumers.size() == 1, "BrgemmCopyB must have only 1 consumer");
        const auto buffer = ov::as_type_ptr<ov::snippets::op::IntermediateMemoryBuffer>(consumers.begin()->get_expr()->get_node());
        OPENVINO_ASSERT(buffer, "BrgemmCopyB consumer must be Buffer");
        return buffer;
    };

    bool modified = false;
    for (auto expr_it = begin; expr_it != end; ++expr_it) {
        const auto& expr = *expr_it;
        if (auto copy_b = ov::as_type_ptr<ov::intel_cpu::BrgemmCopyB>(expr->get_node())) {
            const auto buffer = get_buffer_from_output(expr, 0);
            const auto buffer_shape = copy_b->get_repacking_buffer_shape();
            buffer->set_allocation_size(ov::shape_size(buffer_shape));
            if (copy_b->is_with_compensations()) {
                const auto compensations_buffer = get_buffer_from_output(expr, 1);
                compensations_buffer->set_allocation_size(ov::shape_size(copy_b->get_compensations_buffer_shape()));
            }
            modified = true;
        }
    }
    return modified;
}
