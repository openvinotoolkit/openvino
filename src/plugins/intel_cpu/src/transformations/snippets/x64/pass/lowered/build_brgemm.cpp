// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "build_brgemm.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu_shape.h"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/lowered/loop_info.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/op/brgemm.hpp"
#include "snippets/op/buffer.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/snippets/x64/op/gemm_cpu.hpp"
#include "transformations/snippets/x64/op/brgemm_utils.hpp"
#include "transformations/tpp/x64/op/modifiers.hpp"
#include "utils/general_utils.h"

namespace ov {
namespace intel_cpu {

bool pass::BuildBrgemm::run(snippets::lowered::LinearIR& linear_ir,
                            snippets::lowered::LinearIR::constExprIt begin,
                            snippets::lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::BuildBrgemm")
    bool modified = false;

    for (auto expr_it = begin; expr_it != end; expr_it++) {
        const auto& expr = *expr_it;
        const auto gemm_node = ov::as_type_ptr<GemmCPU>(expr->get_node());
        if (!gemm_node || gemm_node->is_dynamic() || with_repacking(gemm_node->get_type())) {
            continue;
        }
        const auto& loop_manager = linear_ir.get_loop_manager();
        OPENVINO_ASSERT(loop_manager, "GemmCPU node should have a loop manager.");

        const auto loop_ids = expr->get_loop_ids();
        if (loop_ids.empty()) {
            continue;
        }

        const auto& gemm_in0_desc = expr->get_input_port_descriptor(0);
        const auto& gemm_in1_desc = expr->get_input_port_descriptor(1);
        const auto& gemm_out_desc = expr->get_output_port_descriptor(0);

        const auto in0_subtensor = gemm_in0_desc->get_subtensor();
        const auto in1_subtensor = gemm_in1_desc->get_subtensor();
        const auto out_subtensor = gemm_out_desc->get_subtensor();

        // Get innermost loop info
        // TODO: check K-loop
        const auto& inner_loop_info = loop_manager->get_loop_info<snippets::lowered::UnifiedLoopInfo>(loop_ids.front());
        if (inner_loop_info->is_dynamic()) {
            continue;
        }
        auto iter_count = inner_loop_info->get_work_amount() / inner_loop_info->get_increment();

        auto brgemm_node =
            std::make_shared<BrgemmCPU>(expr->get_input_port_connector(0)->get_source().get_expr()->get_node(),
                                        expr->get_input_port_connector(1)->get_source().get_expr()->get_node(),
                                        iter_count,
                                        gemm_node->get_type(),
                                        gemm_node->get_offset_a(),
                                        gemm_node->get_offset_b(),
                                        gemm_node->get_offset_c(),
                                        gemm_in0_desc->get_layout(),
                                        gemm_in1_desc->get_layout(),
                                        gemm_out_desc->get_layout());
        // Replace GemmCPU node with BrgemmCPU
        auto live_regs = expr->get_live_regs();
        snippets::lowered::PortDescriptorUtils::set_port_descriptor(brgemm_node->input(0), in0_subtensor, gemm_in0_desc->get_layout());
        snippets::lowered::PortDescriptorUtils::set_port_descriptor(brgemm_node->input(1), in1_subtensor, gemm_in1_desc->get_layout());
        snippets::lowered::PortDescriptorUtils::set_port_descriptor(brgemm_node->output(0), out_subtensor, gemm_out_desc->get_layout());
        expr_it = linear_ir.replace_with_node({expr}, brgemm_node, expr->get_loop_ids(), linear_ir.find(expr));
        ov::replace_node_update_name(gemm_node, brgemm_node);
        OPENVINO_ASSERT(expr_it != linear_ir.end(), "Failed to replace GemmCPU with BrgemmCPU");

        const auto& updated_expr = *expr_it;
        updated_expr->set_live_regs(std::move(live_regs));

        modified |= true;
    }

    return modified;
}

} // namespace intel_cpu
} // namespace ov
