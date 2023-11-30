// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/init_buffers_default.hpp"

#include "snippets/lowered/pass/allocate_buffers.hpp"
#include "snippets/op/buffer.hpp"
#include "snippets/itt.hpp"


namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

bool InitBuffersDefault::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::InitBuffersDefault");

    size_t id = 0;
    size_t offset = 0;
    for (const auto& expr : linear_ir) {
        const auto op = expr->get_node();
        if (const auto buffer = ov::as_type_ptr<op::Buffer>(op)) {
            AllocateBuffers::set_buffer_offset(expr, offset);
            buffer->set_id(id);

            offset += buffer->get_byte_size();
            id++;
        }
    }

    m_buffer_scratchpad_size = offset;
    return m_buffer_scratchpad_size > 0;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
