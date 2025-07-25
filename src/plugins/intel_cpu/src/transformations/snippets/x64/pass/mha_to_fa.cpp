// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mha_to_fa.hpp"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "snippets/itt.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/x64/op/brgemm_utils.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"
#include "transformations/snippets/x64/op/fa_utils.hpp"
#include "transformations/snippets/x64/op/fa.hpp"

namespace ov::intel_cpu {

pass::MHAToFA::MHAToFA() {
    MATCHER_SCOPE(MHAToFA);
    auto static_shape_single_consumer = [](const ov::Output<ov::Node>& out) {
        return ov::pass::pattern::has_static_shape()(out) && ov::pass::pattern::consumers_count(1)(out);
    };
    const auto matmul0_m = ov::pass::pattern::wrap_type<opset1::MatMul>(static_shape_single_consumer);
    const auto softmax_m =
        ov::pass::pattern::wrap_type<ov::op::v1::Softmax, ov::op::v8::Softmax>({matmul0_m},
                                                                               static_shape_single_consumer);
    const auto matmul1_m = ov::pass::pattern::wrap_type<opset1::MatMul>({softmax_m, ov::pass::pattern::any_input()});

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::MHAToFA")
        const auto& pm = m.get_pattern_value_map();
        const auto matmul0 = as_type_ptr<ov::opset1::MatMul>(pm.at(matmul0_m).get_node_shared_ptr());
        if (matmul0->get_transpose_a()) {
            return false;
        }
        if (matmul0->get_input_element_type(0) != element::Type_t::f32 || matmul0->get_input_element_type(1) != element::Type_t::f32) {
            return false;
        }
        const auto matmul1 = as_type_ptr<ov::opset1::MatMul>(pm.at(matmul1_m).get_node_shared_ptr());
        if (matmul1->get_transpose_a() || matmul1->get_transpose_b()) {
            return false;
        }
        if (matmul1->get_input_element_type(0) != element::Type_t::f32 || matmul1->get_input_element_type(1) != element::Type_t::f32) {
            return false;
        }

        const auto fa_config = fa_utils::FAConfig(ov::element::f32,
                                                  ov::element::f32,
                                                  ov::element::f32,
                                                  matmul0->get_transpose_b());

        const auto brgemm_config = brgemm_utils::BrgemmConfig(fa_config.src_dt(),
                                                              fa_config.wei_dt(),
                                                              fa_config.orig_wei_dt(),
                                                              false,                      // are_wei_constant,
                                                              fa_config.transposed_b());  // BrgemmCopyB::is_transposed(layout_b));
        auto mm0_in1 = matmul0->input_value(1);

        std::shared_ptr<BrgemmCopyB> repack_k = std::make_shared<BrgemmCopyB>(mm0_in1, brgemm_config);

        const auto& q = matmul0->input_value(0);
        const auto& v = matmul1->input_value(1);

        const auto& snippets_fa = std::make_shared<FACPU>(ov::OutputVector{q, repack_k->output(0), v}, fa_config);

        ov::replace_node(matmul1, snippets_fa);
        snippets_fa->set_friendly_name(matmul1->get_friendly_name());
        ov::copy_runtime_info(matmul1, snippets_fa);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(matmul1_m, matcher_name);
    register_matcher(m, callback);
}

}  // namespace ov::intel_cpu