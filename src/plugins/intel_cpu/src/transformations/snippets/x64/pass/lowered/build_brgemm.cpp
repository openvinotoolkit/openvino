    // Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "build_brgemm.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu_shape.h"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/lowered/loop_info.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "snippets/itt.hpp"
#include "snippets/op/brgemm.hpp"
#include "snippets/op/buffer.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/snippets/x64/op/brgemm_utils.hpp"
#include "transformations/tpp/x64/op/modifiers.hpp"
#include "utils/general_utils.h"

namespace ov {
namespace intel_cpu {

bool pass::BuildBrgemm::run(const snippets::lowered::LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::AdjustBrgemmCopyBLoopPorts")
    fprintf(stderr, "BuildBrgemm::run\n");
    bool modified = false;
    for (const auto& expr : linear_ir) {
        fprintf(stderr, "expr: %s\n", expr->get_node()->get_friendly_name().c_str());
        const auto brgemm_node = ov::as_type_ptr<BrgemmCPU>(expr->get_node());
        if (!brgemm_node || brgemm_node->is_dynamic()) {
            continue;
        }
        const auto& loop_manager = linear_ir.get_loop_manager();
        OPENVINO_ASSERT(loop_manager, "BrgemmCPU node should have a loop manager.");

        const auto loop_ids = expr->get_loop_ids();
        if (!loop_ids.empty()) {
            // Get innermost loop info
            // auto loop_expr = loop_manager->get_loop_bounds(linear_ir, loop_ids.back()).first;
            // fprintf(stderr, "Loop bounds: %s\n", loop_expr->get()->get_node()->get_friendly_name().c_str());
            // const auto& inner_loop_info = loop_manager->get_loop_info<snippets::lowered::UnifiedLoopInfo>(loop_ids.front());
            // fprintf(stderr, "work_amount: %ld\n", inner_loop_info->get_work_amount());
            // fprintf(stderr, "increment: %ld\n", inner_loop_info->get_increment());
            // auto iter_count = inner_loop_info->get_work_amount() / inner_loop_info->get_increment();
            // fprintf(stderr, "iter_count: %ld\n", iter_count);
        }

    }

    return modified;
}

} // namespace intel_cpu
} // namespace ov
