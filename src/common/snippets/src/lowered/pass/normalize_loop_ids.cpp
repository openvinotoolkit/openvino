// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/normalize_loop_ids.hpp"

#include "snippets/lowered/loop_manager.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

bool NormalizeLoopIDs::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::NormalizeLoopIDs")
    const auto& loop_manager = linear_ir.get_loop_manager();
    return loop_manager->normalize(linear_ir);
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
