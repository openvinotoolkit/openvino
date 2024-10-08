// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"

#include "brgemm_to_brgemm_cpu.hpp"

#include "snippets/utils/utils.hpp"
#include "snippets/op/brgemm.hpp"
#include "snippets/op/buffer.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_a.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/tpp/x64/op/modifiers.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/matcher.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"
#include "transformations/snippets/x64/op/brgemm_utils.hpp"

#include "cpu_shape.h"
#include "utils/general_utils.h"


namespace ov {
namespace intel_cpu {

using namespace snippets::lowered;

namespace {
template<typename T>
void set_full_port_desc(const T& port) {
    const auto& shape_rank = port.get_partial_shape().size();
    static const std::vector<size_t> full_dim_subtensor(std::min(shape_rank, size_t(2)), ov::snippets::utils::get_full_dim_value());
    PortDescriptorUtils::set_port_descriptor(port, full_dim_subtensor);
}
} // namespace

pass::BrgemmToBrgemmCPU::BrgemmToBrgemmCPU() {
    MATCHER_SCOPE(BrgemmToBrgemmCPU);
    auto is_not_tpp = [](const Output<Node>& out) {
        return !std::dynamic_pointer_cast<const intel_cpu::tpp::modifier::TensorProcessingPrimitive>(out.get_node_shared_ptr());
    };
    auto m_brgemm = ov::pass::pattern::wrap_type<snippets::op::Brgemm>(is_not_tpp);

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::BrgemmToBrgemmCPU")
        const auto node = m.get_match_root();
        const auto brgemm = ov::as_type_ptr<snippets::op::Brgemm>(node);
        const auto brgemm_plugin = ov::as_type_ptr<BrgemmCPU>(node);
        if (!brgemm || brgemm_plugin)
            OPENVINO_THROW("BrgemmCPU cannot be in body before BrgemmToBrgemmCPU pass");

        const auto& brgemm_in0_desc = PortDescriptorUtils::get_port_descriptor_ptr(brgemm->input(0));
        const auto& brgemm_in1_desc = PortDescriptorUtils::get_port_descriptor_ptr(brgemm->input(1));
        const auto& brgemm_out_desc = PortDescriptorUtils::get_port_descriptor_ptr(brgemm->output(0));

        const auto dimsMatMulIn1 = snippets::utils::get_planar_pshape(brgemm->input(1));
        const auto K = ov::snippets::utils::dimension_to_size_t(*++dimsMatMulIn1.rbegin());
        const auto element_type_a = brgemm->get_input_element_type(0);
        const auto element_type_b = brgemm->get_input_element_type(1);

        std::shared_ptr<BrgemmCPU> brgemm_cpu = nullptr;
        std::shared_ptr<BrgemmCopyA> brgemm_copy_a = nullptr;
        std::shared_ptr<BrgemmCopyB> brgemm_copy_b = nullptr;

        auto brgemm_in0 = brgemm->input_value(0);
        auto brgemm_in1 = brgemm->input_value(1);

        auto layout_a = brgemm_in0_desc->get_layout();
        auto layout_b = brgemm_in1_desc->get_layout();
        auto layout_c = brgemm_out_desc->get_layout();

        auto offset_a = brgemm->get_offset_a();
        auto offset_b = brgemm->get_offset_b();
        auto offset_c = brgemm->get_offset_c();

        const bool transpose_b = !layout_b.empty() && layout_b.back() != layout_b.size() - 1;
        const auto brgemm_config = brgemm_utils::BrgemmConfig(element_type_a, element_type_b, K, transpose_b);

        if (brgemm_config.need_copy_a()) {
            brgemm_copy_a = std::make_shared<BrgemmCopyA>(brgemm_in0, brgemm_config, offset_a, 0, layout_a);
            PortDescriptorUtils::set_port_descriptor(brgemm_copy_a->input(0), brgemm_in0_desc->get_subtensor(), layout_a);
            set_full_port_desc(brgemm_copy_a->output(0));

            brgemm_in0 = brgemm_copy_a->output(0);
            layout_a.clear();
            offset_a = 0;
        }

        if (brgemm_config.need_copy_b()) {
            brgemm_copy_b = std::make_shared<BrgemmCopyB>(brgemm_in1, element_type_a, brgemm_config, offset_b, 0, 0, layout_b);
            PortDescriptorUtils::set_port_descriptor(brgemm_copy_b->input(0), brgemm_in1_desc->get_subtensor(), layout_b);
            for (const auto& out : brgemm_copy_b->outputs())
                set_full_port_desc(out);

            brgemm_in1 = brgemm_copy_b->output(0);
            layout_b.clear();
            offset_b = 0;
        }

        if (brgemm_config.need_wsp()) {
            const auto scratch = std::make_shared<snippets::op::Buffer>(ov::Shape{BrgemmCPU::SCRATCH_BYTE_SIZE});
            brgemm_cpu = std::make_shared<BrgemmCPU>(brgemm_in0, brgemm_in1, scratch, brgemm_config,
                                                     offset_a, offset_b, 0, offset_c, layout_a, layout_b, layout_c);

            set_full_port_desc(scratch->output(0));
            set_full_port_desc(brgemm_cpu->input(2));
        } else if (brgemm_config.need_compensations()) {
            OPENVINO_ASSERT(brgemm_copy_b, "Needs to BrgemmCopyB");
            brgemm_cpu = std::make_shared<BrgemmCPU>(brgemm_in0, brgemm_in1, brgemm_copy_b->output(1), brgemm_config,
                                                     offset_a, offset_b, 0, offset_c, layout_a, layout_b, layout_c);
        } else {
            brgemm_cpu = std::make_shared<BrgemmCPU>(brgemm_in0, brgemm_in1, brgemm_config, offset_a, offset_b, offset_c,
                                                     layout_a, layout_b, layout_c);
        }

        brgemm_cpu->set_friendly_name(brgemm->get_friendly_name());
        ov::replace_node(brgemm, brgemm_cpu);

        // need to run validate_and_infer_types manually: either input shapes were updated or
        // output Layout was updated (out shape will be updated in validate_and_infer_types())

        if (brgemm_copy_a) {
            set_full_port_desc(brgemm_cpu->input(0));
            brgemm_copy_a->validate_and_infer_types();
        } else {
            PortDescriptorUtils::set_port_descriptor(brgemm_cpu->input(0), brgemm_in0_desc->get_subtensor(), layout_a);
        }

        if (brgemm_copy_b) {
            set_full_port_desc(brgemm_cpu->input(1));
            brgemm_copy_b->validate_and_infer_types();
        } else {
            PortDescriptorUtils::set_port_descriptor(brgemm_cpu->input(1), brgemm_in1_desc->get_subtensor(), layout_b);
        }

        PortDescriptorUtils::set_port_descriptor(brgemm_cpu->output(0), brgemm_out_desc->get_subtensor(), layout_c);

        brgemm_cpu->validate_and_infer_types();

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(m_brgemm, matcher_name);
    register_matcher(m, callback);
}
} // namespace intel_cpu
} // namespace ov
