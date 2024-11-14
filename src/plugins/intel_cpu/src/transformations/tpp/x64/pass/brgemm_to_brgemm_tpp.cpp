// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"

#include "brgemm_to_brgemm_tpp.hpp"

#include "snippets/utils/utils.hpp"
#include "snippets/op/brgemm.hpp"
#include "transformations/tpp/x64/op/brgemm.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/matcher.hpp"

#include "cpu_shape.h"
#include "utils/general_utils.h"


namespace ov {
namespace intel_cpu {
namespace tpp {
namespace pass {

using namespace snippets::lowered;

bool BrgemmToBrgemmTPP::is_supported_brgemm_configuration(const std::vector<std::vector<size_t>>& layouts,
                                                          const ov::element::TypeVector& precisions) {
    OPENVINO_ASSERT(layouts.size() == 3 && precisions.size() == 3, "snippets::op::Brgemm must have 2 inputs and 1 output");
    const bool supported_layouts = std::all_of(layouts.begin(), layouts.end(), [](const std::vector<size_t>& layout) {
        return layout.empty() || layout.back() == layout.size() - 1;
    });
    const bool supported_precisions = std::all_of(precisions.begin(), precisions.end(), [](const ov::element::Type& et) {
        return et == ov::element::f32;
    });
    return supported_layouts && supported_precisions;
}

BrgemmToBrgemmTPP::BrgemmToBrgemmTPP() {
    MATCHER_SCOPE(BrgemmToBrgemmTPP);

    auto m_brgemm = ov::pass::pattern::wrap_type<snippets::op::Brgemm>();

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::BrgemmToBrgemmTPP")
        const auto node = m.get_match_root();
        const auto brgemm = ov::as_type_ptr<snippets::op::Brgemm>(node);
        if (!brgemm || ov::as_type_ptr<tpp::op::BrgemmTPP>(node))
            OPENVINO_THROW("BrgemmCPU cannot be in body before BrgemmToBrgemmTPP pass");

        if (brgemm->is_dynamic()) {
            return false;
        }

        const auto& brgemm_in0_desc = PortDescriptorUtils::get_port_descriptor_ptr(brgemm->input(0));
        const auto& brgemm_in1_desc = PortDescriptorUtils::get_port_descriptor_ptr(brgemm->input(1));
        const auto& brgemm_out_desc = PortDescriptorUtils::get_port_descriptor_ptr(brgemm->output(0));

        const auto& layout_a = brgemm_in0_desc->get_layout();
        const auto& layout_b = brgemm_in1_desc->get_layout();
        const auto& layout_c = brgemm_out_desc->get_layout();

        const auto& precision_a = brgemm->get_input_element_type(0);
        const auto& precision_b = brgemm->get_input_element_type(1);
        const auto& precision_c = brgemm->get_output_element_type(0);

        if (!is_supported_brgemm_configuration({layout_a, layout_b, layout_c}, {precision_a, precision_b, precision_c}))
            return false;

        const auto dimsMatMulIn0 = snippets::utils::get_planar_pshape(brgemm->input(0)).get_shape();
        const auto dimsMatMulIn1 = snippets::utils::get_planar_pshape(brgemm->input(1)).get_shape();

        const auto offset_a = brgemm->get_offset_a();
        const auto offset_b = brgemm->get_offset_b();
        const auto offset_c = brgemm->get_offset_c();

        std::shared_ptr<tpp::op::BrgemmTPP> brgemm_tpp = nullptr;
        if (precision_a == ov::element::f32) {
            brgemm_tpp = std::make_shared<tpp::op::BrgemmTPP>(brgemm->input_value(0),
                                                              brgemm->input_value(1),
                                                              offset_a, offset_b, offset_c,
                                                              layout_a, layout_b, layout_c);
        }
        OPENVINO_ASSERT(brgemm_tpp, "Failed to create BrgemmTPP node in the BrgemmToBrgemmTPP pass");
        brgemm_tpp->set_friendly_name(brgemm->get_friendly_name());
        ov::replace_node(brgemm, brgemm_tpp);

        // Set FULL_DIM tensors on ports to avoid automatic loop markup (blocked loops will be inserted in a separate transformation)
        PortDescriptorUtils::set_port_descriptor(brgemm_tpp->input(0), brgemm_in0_desc->get_subtensor(), brgemm_in0_desc->get_layout());
        PortDescriptorUtils::set_port_descriptor(brgemm_tpp->input(1), brgemm_in1_desc->get_subtensor(), brgemm_in1_desc->get_layout());
        PortDescriptorUtils::set_port_descriptor(brgemm_tpp->output(0), brgemm_out_desc->get_subtensor(), brgemm_out_desc->get_layout());

        // need to run validate_and_infer_types manually: either input shapes were updated or
        // output Layout was updated (out shape will be updated in validate_and_infer_types())
        brgemm_tpp->validate_and_infer_types();

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(m_brgemm, matcher_name);
    register_matcher(m, callback);
}
} // namespace pass
} // namespace tpp
} // namespace intel_cpu
} // namespace ov
