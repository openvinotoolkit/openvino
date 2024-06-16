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
    for (auto expr_it = begin; expr_it != end; ++expr_it) {
        const auto& expr = *expr_it;
        const auto op = expr->get_node();
        if (const auto buffer = ov::as_type_ptr<op::Buffer>(op)) {
            buffer->set_reg_group(idx);
            buffer->set_cluster_id(idx);

            if (!buffer->is_defined()) {
                buffer->set_offset(utils::get_dynamic_value<size_t>());
            } else {
                buffer->set_offset(offset);
                offset += buffer->get_byte_size();
            }
            idx++;
        }
    }

    m_buffer_scratchpad_size = offset;
    return m_buffer_scratchpad_size > 0;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
