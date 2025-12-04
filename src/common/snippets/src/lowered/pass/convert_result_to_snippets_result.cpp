// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/convert_result_to_snippets_result.hpp"

#include <memory>

#include "openvino/core/node_vector.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/result.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/op/result.hpp"

namespace ov::snippets::lowered::pass {

bool ConvertResultToSnippetsResult::run(LinearIR& linear_ir,
                                        lowered::LinearIR::constExprIt begin,
                                        lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::ConvertResultToSnippetsResult")
    bool modified = false;

    for (auto expr_it = begin; expr_it != end; expr_it++) {
        const auto& expr = *expr_it;
        const auto& op = expr->get_node();
        if (const auto result = ov::as_type_ptr<ov::op::v0::Result>(op)) {
            OutputVector new_op_inputs = {expr->get_input_expr_ptr(0)->get_node()};
            const auto snippets_result = std::make_shared<snippets::op::Result>(new_op_inputs);
            expr_it = linear_ir.replace_with_node({expr}, snippets_result);
            modified |= true;
        }
    }
    return modified;
}

}  // namespace ov::snippets::lowered::pass
