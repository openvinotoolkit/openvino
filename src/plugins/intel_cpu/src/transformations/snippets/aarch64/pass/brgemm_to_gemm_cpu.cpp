// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_to_gemm_cpu.hpp"

#include "cpu_shape.h"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "snippets/itt.hpp"
#include "snippets/op/brgemm.hpp"
#include "snippets/op/buffer.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/aarch64/op/gemm_copy_b.hpp"
#include "transformations/snippets/aarch64/op/gemm_cpu.hpp"
#include "transformations/tpp/common/op/modifiers.hpp"
#include "utils/general_utils.h"
// #include "openvino/pass/manager.hpp"

namespace ov::intel_cpu {

using namespace snippets::lowered;

namespace {
template <typename T>
void set_full_port_desc(const T& port) {
    const auto& shape_rank = port.get_partial_shape().size();
    const std::vector<size_t> full_dim_subtensor(std::min(shape_rank, static_cast<size_t>(2)),
                                                 ov::snippets::utils::get_full_dim_value());
    PortDescriptorUtils::set_port_descriptor(port, full_dim_subtensor);
}
}  // namespace

pass::BrgemmToGemmCPU::BrgemmToGemmCPU() {
    MATCHER_SCOPE(BrgemmToGemmCPU);
    auto is_not_tpp = [](const Output<Node>& out) {
        return !std::dynamic_pointer_cast<const intel_cpu::tpp::modifier::TensorProcessingPrimitive>(
            out.get_node_shared_ptr());
    };
    auto m_brgemm = ov::pass::pattern::wrap_type<snippets::op::Brgemm>(is_not_tpp);

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::BrgemmToGemmCPU")
        const auto node = m.get_match_root();
        const auto brgemm = ov::as_type_ptr<snippets::op::Brgemm>(node);
        const auto brgemm_plugin = ov::as_type_ptr<aarch64::GemmCPU>(node);
        if (!brgemm || brgemm_plugin) {
            OPENVINO_THROW("GemmCPU cannot be in body before BrgemmToGemmCPU pass");
        }

        const auto& brgemm_in0_desc = PortDescriptorUtils::get_port_descriptor_ptr(brgemm->input(0));
        const auto& brgemm_in1_desc = PortDescriptorUtils::get_port_descriptor_ptr(brgemm->input(1));
        const auto& brgemm_out_desc = PortDescriptorUtils::get_port_descriptor_ptr(brgemm->output(0));

        const auto dimsMatMulIn0 = snippets::utils::get_planar_pshape(brgemm->input(0));
        const auto dimsMatMulIn1 = snippets::utils::get_planar_pshape(brgemm->input(1));

        const auto& layout_a = brgemm_in0_desc->get_layout();
        const auto& layout_b = brgemm_in1_desc->get_layout();
        const auto& layout_c = brgemm_out_desc->get_layout();

        const auto element_type_a = brgemm->get_input_element_type(0);
        // const bool transpose_b = !layout_b.empty() && layout_b.back() != layout_b.size() - 1;
        const auto offset_a = brgemm->get_offset_a();
        const auto offset_b = brgemm->get_offset_b();
        const auto offset_c = brgemm->get_offset_c();

        auto gemm_repacking =
            std::make_shared<aarch64::GemmCopyB>(brgemm->input_value(1), element_type_a, offset_b, 0, layout_b);
        PortDescriptorUtils::set_port_descriptor(gemm_repacking->input(0), brgemm_in1_desc->get_subtensor(), layout_b);
        for (const auto& output : gemm_repacking->outputs()) {
            set_full_port_desc(output);
        }

        std::shared_ptr<aarch64::GemmCPU> gemm_cpu = std::make_shared<aarch64::GemmCPU>(brgemm->input_value(0),
                                                                                        gemm_repacking->output(0),
                                                                                        offset_a,
                                                                                        0,
                                                                                        offset_c,
                                                                                        layout_a,
                                                                                        std::vector<size_t>{},
                                                                                        layout_c);

        gemm_cpu->set_friendly_name(brgemm->get_friendly_name());
        ov::replace_node(brgemm, gemm_cpu);

        // Transfer ports
        PortDescriptorUtils::set_port_descriptor(gemm_cpu->input(0), brgemm_in0_desc->get_subtensor(), layout_a);
        set_full_port_desc(gemm_cpu->input(1));
        PortDescriptorUtils::set_port_descriptor(gemm_cpu->output(0), brgemm_out_desc->get_subtensor(), layout_c);

        // need to run validate_and_infer_types manually: either input shapes were updated or
        // output Layout was updated (out shape will be updated in validate_and_infer_types())
        gemm_repacking->validate_and_infer_types();
        gemm_cpu->validate_and_infer_types();

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(m_brgemm, matcher_name);
    register_matcher(m, callback);
}
}  // namespace ov::intel_cpu
