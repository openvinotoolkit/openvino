// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/init_buffers_default.hpp"

#include "snippets/op/buffer.hpp"
#include "snippets/itt.hpp"


namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

bool InitBuffersDefault::run(lowered::LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::InitBuffersDefault");

    size_t idx = 0;
    size_t offset = 0;
    for (const auto& buffer_expr : linear_ir.get_buffers()) {
        buffer_expr->set_reg_group(idx);
        buffer_expr->set_cluster_id(idx);

        if (!buffer_expr->is_defined()) {
            buffer_expr->set_offset(utils::get_dynamic_value<size_t>());
        } else {
            buffer_expr->set_offset(offset);
            offset += buffer_expr->get_byte_size();
        }
        idx++;
    }
    for (auto expr_it = begin; expr_it != end; ++expr_it) {
        const auto& expr = *expr_it;
        const auto op = expr->get_node();
        if (const auto inplace_buffer = ov::as_type_ptr<op::InplaceMemoryBuffer>(op)) {
            const auto& inplace_from_buffer = ov::as_type_ptr<op::Buffer>(inplace_buffer->get_inplace_from());
            if (inplace_from_buffer) {
                inplace_buffer->set_reg_group(inplace_from_buffer->get_reg_group());
                inplace_buffer->set_cluster_id(inplace_from_buffer->get_cluster_id());
                if (!inplace_buffer->is_defined()) {
                    inplace_buffer->set_offset(utils::get_dynamic_value<size_t>());
                } else {
                    inplace_buffer->set_offset(inplace_from_buffer->get_offset());
                }
            }
        }
    }

    m_buffer_scratchpad_size = offset;
    return m_buffer_scratchpad_size > 0;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
