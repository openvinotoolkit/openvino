// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"

#include "brgemm_to_brgemm_cpu.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"

#include "ngraph/rt_info.hpp"
#include "ngraph/pattern/op/wrap_type.hpp"

#include <cpu/x64/cpu_isa_traits.hpp>

#include "cpu_shape.h"
#include "utils/general_utils.h"


namespace ov {
namespace intel_cpu {

pass::BrgemmToBrgemmCPU::BrgemmToBrgemmCPU() {
    MATCHER_SCOPE(BrgemmToBrgemmCPU);

    auto m_brgemm = ngraph::pattern::wrap_type<ngraph::snippets::op::Brgemm>();

    auto callback = [=](ngraph::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::BrgemmToBrgemmCPU")
        const auto node = m.get_match_root();
        const auto brgemm = ov::as_type_ptr<ngraph::snippets::op::Brgemm>(node);
        const auto brgemm_plugin = ov::as_type_ptr<BrgemmCPU>(node);
        if (!brgemm || brgemm_plugin)
            throw ov::Exception("BrgemmCPU cannot be in body before BrgemmToBrgemmCPU pass");

        if (brgemm->is_dynamic()) {
            return false;
        }

        const auto dimsMatMulIn0 = ngraph::snippets::utils::get_port_planar_shape(brgemm->input_value(0)).get_shape();
        const auto dimsMatMulIn1 = ngraph::snippets::utils::get_port_planar_shape(brgemm->input_value(1)).get_shape();

        const auto K = *dimsMatMulIn0.rbegin();
        const auto N = *dimsMatMulIn1.rbegin();

        const auto element_type_a = brgemm->get_input_element_type(0);
        const auto brgemmVNNIFactor = 4 / element_type_a.size();
        const bool isAMXSupported = dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx);
        const bool with_amx = isAMXSupported && element_type_a != ov::element::f32 && (K % brgemmVNNIFactor == 0) && (N % brgemmVNNIFactor == 0);
        const bool with_comp = element_type_a == ov::element::i8 && !with_amx;

        const auto offset_a = brgemm->get_offset_a();
        const auto offset_b = brgemm->get_offset_b();
        const auto offset_c = brgemm->get_offset_c();

        std::shared_ptr<ov::Node> brgemm_cpu = nullptr;
        if (element_type_a == ov::element::f32) {
            brgemm_cpu = std::make_shared<BrgemmCPU>(brgemm->input_value(0), brgemm->input_value(1), BrgemmCPU::Type::Floating,
                                                     offset_a, offset_b, offset_c);
        } else {
            const auto layoutIn1 = ngraph::snippets::utils::get_node_output_layout(brgemm->input_value(1).get_node_shared_ptr());
            const auto copy_b_type = with_comp ? BrgemmCopyB::WithCompensations : BrgemmCopyB::OnlyRepacking;
            const auto brgemmRepackIn1 = std::make_shared<BrgemmCopyB>(brgemm->input_value(1), element_type_a, copy_b_type, offset_b);
            const auto buffer = std::make_shared<ngraph::snippets::op::Buffer>(brgemmRepackIn1->output(0));

            if (with_amx) {
                const auto scratch = std::make_shared<ngraph::snippets::op::Buffer>(ov::Shape{BrgemmCPU::SCRATCH_BYTE_SIZE});
                brgemm_cpu = std::make_shared<BrgemmCPU>(brgemm->input_value(0), buffer, scratch, BrgemmCPU::Type::AMX,
                                                         offset_a, offset_b, offset_c);
            } else if (with_comp) {
                const auto scratch = std::make_shared<ngraph::snippets::op::Buffer>(brgemmRepackIn1->output(1));
                brgemm_cpu = std::make_shared<BrgemmCPU>(brgemm->input_value(0), buffer, scratch, BrgemmCPU::Type::WithCompensations,
                                                         offset_a, offset_b, offset_c);
            } else if (one_of(element_type_a, ov::element::u8, ov::element::bf16)) {
                brgemm_cpu = std::make_shared<BrgemmCPU>(brgemm->input_value(0), buffer, BrgemmCPU::Type::WithDataRepacking,
                                                         offset_a, offset_b, offset_c);
            } else {
                IE_THROW() << "Invalid configuration for BRGEMM CPU";
            }
        }

        brgemm_cpu->set_friendly_name(brgemm->get_friendly_name());
        ngraph::snippets::utils::set_output_layout(brgemm_cpu->output(0), ngraph::snippets::utils::get_node_output_layout(brgemm));
        ngraph::copy_runtime_info(brgemm, brgemm_cpu);
        ngraph::replace_node(brgemm, brgemm_cpu);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(m_brgemm, matcher_name);
    register_matcher(m, callback);
}
} // namespace intel_cpu
} // namespace ov
