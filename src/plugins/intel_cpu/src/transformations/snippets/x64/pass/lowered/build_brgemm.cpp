// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "build_brgemm.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu_shape.h"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/loop_info.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/op/brgemm.hpp"
#include "snippets/op/buffer.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/snippets/x64/op/brgemm_utils.hpp"
#include "transformations/snippets/x64/op/gemm_cpu.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu {

bool pass::BuildBrgemm::run(snippets::lowered::LinearIR& linear_ir,
                            snippets::lowered::LinearIR::constExprIt begin,
                            snippets::lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::BuildBrgemm")
    bool modified = false;

    for (auto expr_it = begin; expr_it != end; expr_it++) {
        const auto& expr = *expr_it;
        const auto gemm_node = ov::as_type_ptr<GemmCPU>(expr->get_node());
        if (!gemm_node || gemm_node->is_dynamic() || with_compensations(gemm_node->get_type())) {
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

        // Get innermost loop info
        const auto& inner_loop_info = loop_manager->get_loop_info<snippets::lowered::UnifiedLoopInfo>(loop_ids.back());
        if (inner_loop_info->is_dynamic()) {
            continue;
        }

        const auto& in_ports = inner_loop_info->get_input_ports();
        const auto& out_ports = inner_loop_info->get_output_ports();
        if (!(in_ports.size() >= 2 && in_ports.front().is_processed() && in_ports.front().get_dim_idx() == 0 &&
              in_ports.back().is_processed() && in_ports.back().get_dim_idx() == 1 && out_ports.size() == 1 &&
              !out_ports.front().is_processed())) {
            continue;
        }

        if (inner_loop_info->get_work_amount() % inner_loop_info->get_increment() != 0) {
            continue;
        }
        auto iter_count = inner_loop_info->get_work_amount() / inner_loop_info->get_increment();

        std::shared_ptr<BrgemmCPU> brgemm_node;
        if (with_scratchpad(gemm_node->get_type())) {
            OPENVINO_ASSERT(expr->get_input_port_connectors().size(),
                            "GemmCPU expects 3 inputs with input precisions i8|i8 and bf16|bf16 on AMX system");
            brgemm_node =
                std::make_shared<BrgemmCPU>(expr->get_input_port_connector(0)->get_source().get_expr()->get_node(),
                                            expr->get_input_port_connector(1)->get_source().get_expr()->get_node(),
                                            expr->get_input_port_connector(2)->get_source().get_expr()->get_node(),
                                            iter_count,
                                            gemm_node->get_type(),
                                            gemm_node->get_offset_a(),
                                            gemm_node->get_offset_b(),
                                            gemm_node->get_offset_scratch(),
                                            gemm_node->get_offset_c(),
                                            gemm_in0_desc->get_layout(),
                                            gemm_in1_desc->get_layout(),
                                            gemm_out_desc->get_layout());
        } else {
            OPENVINO_ASSERT(expr->get_input_port_connectors().size() == 2,
                            "GemmCPU expects 2 inputs in cases, when input precisions are f32|f32, u8|i8 or bf16|bf16 "
                            "(non-AMX system)");
            brgemm_node =
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
        }

        auto old_work_amount = inner_loop_info->get_work_amount();
        auto new_increment = old_work_amount * iter_count;
        inner_loop_info->set_increment(new_increment);

        // Replace GemmCPU node with BrgemmCPU
        auto live_regs = expr->get_live_regs();

        auto in0_subtensor = gemm_in0_desc->get_subtensor();
        auto in1_subtensor = gemm_in1_desc->get_subtensor();
        const auto out_subtensor = gemm_out_desc->get_subtensor();

        in0_subtensor[1] *= iter_count;
        in1_subtensor[0] *= iter_count;
        snippets::lowered::PortDescriptorUtils::set_port_descriptor(brgemm_node->input(0),
                                                                    in0_subtensor,
                                                                    gemm_in0_desc->get_layout());
        snippets::lowered::PortDescriptorUtils::set_port_descriptor(brgemm_node->input(1),
                                                                    in1_subtensor,
                                                                    gemm_in1_desc->get_layout());
        if (with_amx(gemm_node->get_type()) || with_compensations(gemm_node->get_type())) {
            const auto& gemm_in2_desc = expr->get_input_port_descriptor(2);
            const auto in2_subtensor = gemm_in2_desc->get_subtensor();
            snippets::lowered::PortDescriptorUtils::set_port_descriptor(brgemm_node->input(2),
                                                                        in2_subtensor,
                                                                        gemm_in2_desc->get_layout());
        }
        snippets::lowered::PortDescriptorUtils::set_port_descriptor(brgemm_node->output(0),
                                                                    out_subtensor,
                                                                    gemm_out_desc->get_layout());

        expr_it = linear_ir.replace_with_node({expr}, brgemm_node, expr->get_loop_ids(), linear_ir.find(expr));
        ov::replace_node_update_name(gemm_node, brgemm_node);
        brgemm_node->set_friendly_name(gemm_node->get_friendly_name());
        brgemm_node->get_rt_info() = gemm_node->get_rt_info();
        ov::replace_node(gemm_node, brgemm_node);
        OPENVINO_ASSERT(expr_it != linear_ir.end(), "Failed to replace GemmCPU with BrgemmCPU");

        const auto& updated_expr = *expr_it;
        updated_expr->set_live_regs(std::move(live_regs));

        modified |= true;
    }

    return modified;
}

}  // namespace ov::intel_cpu
