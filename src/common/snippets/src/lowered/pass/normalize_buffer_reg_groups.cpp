// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/normalize_buffer_reg_groups.hpp"

#include <cstddef>
#include <map>

#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"

namespace ov::snippets::lowered::pass {

bool NormalizeBufferRegisterGroups::run(lowered::LinearIR& linear_ir,
                                        [[maybe_unused]] lowered::LinearIR::constExprIt begin,
                                        [[maybe_unused]] lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::NormalizeBufferRegisterGroups");

    // [ original Buffer reg group -> normalized ]
    std::map<size_t, size_t> buffer_reg_groups;
    for (const auto& buffer_expr : linear_ir.get_buffers()) {
        const auto group = buffer_expr->get_reg_group();
        if (buffer_reg_groups.count(group) == 0) {
            const auto new_id = buffer_reg_groups.size();
            buffer_reg_groups[group] = new_id;
        }
        buffer_expr->set_reg_group(buffer_reg_groups[group]);
    }
    return !buffer_reg_groups.empty();
}

}  // namespace ov::snippets::lowered::pass
