// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/normalize_buffer_ids.hpp"

#include "snippets/op/buffer.hpp"
#include "snippets/itt.hpp"


namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

bool NormalizeBufferIDs::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::NormalizeBufferIDs");

    // [ original Buffer ID -> normalized ]
    std::map<size_t, size_t> buffer_ids;
    for (const auto& expr : linear_ir) {
        const auto op = expr->get_node();
        if (const auto buffer = ov::as_type_ptr<op::Buffer>(op)) {
            const auto buffer_id = buffer->get_id();
            if (buffer_ids.count(buffer_id) == 0) {
                const auto new_id = buffer_ids.size();
                buffer_ids[buffer_id] = new_id;
            }
            buffer->set_id(buffer_ids[buffer_id]);
        }
    }
    return buffer_ids.size();
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
