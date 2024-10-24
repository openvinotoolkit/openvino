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
        if (const auto& buffer_expr = ov::as_type_ptr<BufferExpression>(expr)) {
            if (const auto& inplace_from = buffer_expr->get_inplace_node()) {
                if (ov::as_type_ptr<op::Buffer>(inplace_from)) {
                    const auto& inplace_from_expr = std::find_if(begin, end, [inplace_from](const ExpressionPtr& expr) {
                        return expr->get_node() == inplace_from;
                    });
                    buffer_expr->set_reg_group(ov::as_type_ptr<BufferExpression>(*inplace_from_expr)->get_reg_group());
                    buffer_expr->set_cluster_id(ov::as_type_ptr<BufferExpression>(*inplace_from_expr)->get_cluster_id());
                    if (!buffer_expr->is_defined()) {
                        buffer_expr->set_offset(utils::get_dynamic_value<size_t>());
                    } else {
                        buffer_expr->set_offset(ov::as_type_ptr<BufferExpression>(*inplace_from_expr)->get_offset());
                    }
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
