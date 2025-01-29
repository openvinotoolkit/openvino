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

bool pass::BuildBrgemm::run(const snippets::lowered::LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::AdjustBrgemmCopyBLoopPorts")
    bool modified = false;
    for (const auto& expr : linear_ir) {
        const auto gemm_node = ov::as_type_ptr<GemmCPU>(expr->get_node());
        if (!gemm_node || gemm_node->is_dynamic()) {
            continue;
        }
        const auto& loop_manager = linear_ir.get_loop_manager();
        OPENVINO_ASSERT(loop_manager, "GemmCPU node should have a loop manager.");

        const auto loop_ids = expr->get_loop_ids();
        if (loop_ids.empty()) {
            continue;
        }

        const auto& gemm_in0_desc = snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(gemm_node->input(0));
        const auto& gemm_in1_desc = snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(gemm_node->input(1));
        const auto& gemm_out_desc = snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(gemm_node->output(0));

        // Get innermost loop info
        auto loop_expr = loop_manager->get_loop_bounds(linear_ir, loop_ids.back()).first;
        const auto& inner_loop_info = loop_manager->get_loop_info<snippets::lowered::UnifiedLoopInfo>(loop_ids.front());
        auto iter_count = inner_loop_info->get_work_amount() / inner_loop_info->get_increment();
        auto brgemm_node = std::make_shared<BrgemmCPU>(gemm_node->input_value(0),
                                                       gemm_node->input_value(1),
                                                       iter_count,
                                                       gemm_node->get_type(),
                                                       gemm_node->get_offset_a(),
                                                       gemm_node->get_offset_b(),
                                                       gemm_node->get_offset_c(),
                                                       gemm_in0_desc->get_layout(),
                                                       gemm_in1_desc->get_layout(),
                                                       gemm_out_desc->get_layout());
        // TODO: replace node

        // Transfer ports
        snippets::lowered::PortDescriptorUtils::set_port_descriptor(gemm_node->input(0), gemm_in0_desc->get_subtensor(), gemm_in0_desc->get_layout());
        snippets::lowered::PortDescriptorUtils::set_port_descriptor(gemm_node->input(1), gemm_in1_desc->get_subtensor(), gemm_in1_desc->get_layout());
        snippets::lowered::PortDescriptorUtils::set_port_descriptor(gemm_node->output(0), gemm_out_desc->get_subtensor(), gemm_out_desc->get_layout());

        // need to run validate_and_infer_types manually: either input shapes were updated or
        // output Layout was updated (out shape will be updated in validate_and_infer_types())
        gemm_node->validate_and_infer_types();
        brgemm_node->validate_and_infer_types();

        const auto& inputs = expr->get_node()->inputs();
    }

    return modified;
}

} // namespace intel_cpu
} // namespace ov
