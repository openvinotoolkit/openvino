    // Copyright (C) 2023 Intel Corporation
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

    fprintf(stderr, "Dumping Linear IR <before>:\n");
    for (auto it = begin; it != end; ++it) {
        const auto& expr = *it;
        fprintf(stderr, "%s\n", expr->get_node()->get_friendly_name().c_str());
    }
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

        // TODO: get input port descriptor
        const auto& gemm_in0_desc = expr->get_input_port_descriptor(0);
        const auto& gemm_in1_desc = expr->get_input_port_descriptor(1);
        const auto& gemm_out_desc = expr->get_output_port_descriptor(0);

        // const auto& interm_connector = expr->get_input_port_connector(0);
        // const auto gemm_expr = interm_connector->get_source().get_expr();

        // Get innermost loop info
        // TODO: check K-loop
        const auto& inner_loop_info = loop_manager->get_loop_info<snippets::lowered::UnifiedLoopInfo>(loop_ids.front());
        fprintf(stderr, "inner_loop_info for loop id %zu (inputs count: %zu):\n", loop_ids.front(), inner_loop_info->get_input_ports_info().size());
        for (size_t i = 0; i < inner_loop_info->get_input_ports_info().size(); ++i) {
            fprintf(stderr, "Input port %zu is_processed: %d\n", i, inner_loop_info->get_input_ports_info()[i].port.is_processed());
        }
        // fprintf(stderr, "Output port 0 is_processed: %d\n", inner_loop_info->get_output_ports_info()[1].port.is_processed());
        if (inner_loop_info->get_work_amount() % inner_loop_info->get_increment() != 0) {
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
        expr_it = linear_ir.replace_with_node({expr}, brgemm_node, expr->get_loop_ids(), linear_ir.find(expr));
        expr_it->get()->set_live_regs(std::move(live_regs));
        const auto loop_ids2 = (*expr_it)->get_loop_ids();
        const auto& inner_loop_info2 = loop_manager->get_loop_info<snippets::lowered::UnifiedLoopInfo>(loop_ids2.front());
        fprintf(stderr, "inner_loop_info2 for loop id %zu (inputs count: %zu):\n", loop_ids2.front(), inner_loop_info2->get_input_ports_info().size());
        for (size_t i = 0; i < inner_loop_info2->get_input_ports_info().size(); ++i) {
            fprintf(stderr, "Input port %zu is_processed: %d\n", i, inner_loop_info2->get_input_ports_info()[i].port.is_processed());
        }

        modified |= true;
    }
    fprintf(stderr, "Dumping Linear IR <after>:\n");
    for (auto it = begin; it != end; ++it) {
        const auto& expr = *it;
        fprintf(stderr, "%s\n", expr->get_node()->get_friendly_name().c_str());
    }

    return modified;
}

} // namespace intel_cpu
} // namespace ov
