// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"

#include "brgemm_to_brgemm_tpp.hpp"

#include "snippets/utils.hpp"
#include "snippets/op/brgemm.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"
#include "transformations/snippets/tpp/op/brgemm.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/matcher.hpp"

#include <cpu/x64/cpu_isa_traits.hpp>

#include "cpu_shape.h"
#include "utils/general_utils.h"


namespace ov {
namespace intel_cpu {
namespace tpp {
namespace pass {

using namespace snippets::lowered;

namespace {
template<typename T, typename... Args>
void set_port_desc(const T& port, Args... params) {
    PortDescriptorUtils::set_port_descriptor_ptr(port, std::make_shared<PortDescriptor>(params...));
}
} // namespace

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

        const auto dimsMatMulIn0 = snippets::utils::get_planar_pshape(brgemm->input(0)).get_shape();
        const auto dimsMatMulIn1 = snippets::utils::get_planar_pshape(brgemm->input(1)).get_shape();

        const auto M = *++dimsMatMulIn0.rbegin();
        const auto K = *dimsMatMulIn0.rbegin();
        const auto N = *dimsMatMulIn1.rbegin();

        const auto element_type_a = brgemm->get_input_element_type(0);
        const auto element_type_b = brgemm->get_input_element_type(1);

        const auto offset_a = brgemm->get_offset_a();
        const auto offset_b = brgemm->get_offset_b();
        const auto offset_c = brgemm->get_offset_c();

        std::shared_ptr<tpp::op::BrgemmTPP> brgemm_tpp = nullptr;
        if (element_type_a == ov::element::f32) {
            brgemm_tpp = std::make_shared<tpp::op::BrgemmTPP>(brgemm->input_value(0),
                                                              brgemm->input_value(1),
                                                              offset_a, offset_b, offset_c,
                                                              brgemm_in0_desc->get_layout(),
                                                              brgemm_in1_desc->get_layout(),
                                                              brgemm_out_desc->get_layout());
        }
        OPENVINO_ASSERT(brgemm_tpp, "DEBUG ASSERT: FAILED TO CREATE BrgemmTPP in the BrgemmToBrgemmTPP pass");
        // Set blocking params
        // Ticket: 113745
        // TODO: extend block size selection heuristics
        auto get_block_size_m = [](const size_t M) {
            return 32;
        };
        auto get_block_size_k = [=](const size_t K) {
            if (element_type_b != ov::element::f32)
                return K;
            return K > 1024 ? 1024 : K > 512 ? 512 : K;
        };
        auto get_block_size_n = [=](const size_t N) {
            return element_type_b != ov::element::f32 ? N : 64;
        };

        brgemm_tpp->set_m_block_size(get_block_size_m(M));
        brgemm_tpp->set_k_block_size(get_block_size_k(K));
        brgemm_tpp->set_n_block_size(get_block_size_n(N));

        brgemm_tpp->set_friendly_name(brgemm->get_friendly_name());
        ov::replace_node(brgemm, brgemm_tpp);

        // Set FULL_DIM tensors on ports to avoid automatic loop markup (blocked loops will be inserted in a separate transformation)
        set_port_desc(brgemm_tpp->input(0), brgemm_in0_desc->get_shape(), brgemm_in0_desc->get_subtensor(), brgemm_in0_desc->get_layout());
        set_port_desc(brgemm_tpp->input(1), brgemm_in1_desc->get_shape(), brgemm_in1_desc->get_subtensor(), brgemm_in1_desc->get_layout());
        set_port_desc(brgemm_tpp->output(0), brgemm_out_desc->get_shape(), brgemm_out_desc->get_subtensor(), brgemm_out_desc->get_layout());

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
