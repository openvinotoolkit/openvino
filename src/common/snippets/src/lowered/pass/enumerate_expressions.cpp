// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/enumerate_expressions.hpp"

#include "snippets/itt.hpp"


namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

bool EnumerateExpressions::run(LinearIR& linear_ir) {
    linear_ir.enumerate_expressions();
    return true;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
