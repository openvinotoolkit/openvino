// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/enumerate_expressions.hpp"

#include "snippets/pass/tokenization.hpp"
#include "snippets/itt.hpp"


namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

bool EnumerateExpressions::run(LinearIR& linear_ir) {
    // [113536]: Temporary solution is reusing of topological order from tokenization.
    //           Need to add execution order of Expression support
    int64_t order = 0;
    for (const auto& expr : linear_ir) {
        ov::snippets::pass::SetTopologicalOrder(expr->get_node(), order++);
    }
    return order > 0;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
