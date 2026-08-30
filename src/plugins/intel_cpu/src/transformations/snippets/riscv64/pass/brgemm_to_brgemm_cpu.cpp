// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_to_brgemm_cpu.hpp"

#include <memory>
#include <vector>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/op/brgemm.hpp"
#include "transformations/snippets/riscv64/op/brgemm_cpu.hpp"

namespace ov::intel_cpu {

using namespace snippets::lowered;
using PortDescriptor = ov::snippets::modifier::MemoryAccess::PortDescriptor;

bool pass::BrgemmToBrgemmCPU::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(BrgemmToBrgemmCPU);
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::BrgemmToBrgemmCPU")

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(ov::pass::pattern::wrap_type<snippets::op::Brgemm>(),
                                                                "BrgemmToBrgemmCPU");

    bool status = false;
    for (const auto& node : model->get_ordered_ops()) {
        if (!matcher->match(node)) {
            continue;
        }

        const auto brgemm = ov::as_type_ptr<snippets::op::Brgemm>(matcher->get_match_root());
        OPENVINO_ASSERT(brgemm && !ov::is_type<BrgemmCPU>(brgemm),
                        "BrgemmCPU cannot be in the body before BrgemmToBrgemmCPU");

        const auto& in0_desc = PortDescriptorUtils::get_port_descriptor_ptr(brgemm->input(0));
        const auto& in1_desc = PortDescriptorUtils::get_port_descriptor_ptr(brgemm->input(1));
        const auto& out_desc = PortDescriptorUtils::get_port_descriptor_ptr(brgemm->output(0));
        const auto& layout_a = in0_desc->get_layout();
        const auto& layout_b = in1_desc->get_layout();
        const auto& layout_c = out_desc->get_layout();

        const auto brgemm_cpu = std::make_shared<BrgemmCPU>(brgemm->input_value(0),
                                                            brgemm->input_value(1),
                                                            PortDescriptor{0, brgemm->get_offset_a()},
                                                            PortDescriptor{0, brgemm->get_offset_b()},
                                                            PortDescriptor{0, brgemm->get_offset_c()},
                                                            layout_a,
                                                            layout_b,
                                                            layout_c);

        PortDescriptorUtils::set_port_descriptor(brgemm_cpu->input(0), in0_desc->get_subtensor(), layout_a);
        PortDescriptorUtils::set_port_descriptor(brgemm_cpu->input(1), in1_desc->get_subtensor(), layout_b);
        PortDescriptorUtils::set_port_descriptor(brgemm_cpu->output(0), out_desc->get_subtensor(), layout_c);
        brgemm_cpu->validate_and_infer_types();
        brgemm_cpu->set_friendly_name(brgemm->get_friendly_name());
        ov::copy_runtime_info(brgemm, brgemm_cpu);
        ov::replace_node(brgemm, brgemm_cpu);
        status = true;
    }

    return status;
}

}  // namespace ov::intel_cpu
