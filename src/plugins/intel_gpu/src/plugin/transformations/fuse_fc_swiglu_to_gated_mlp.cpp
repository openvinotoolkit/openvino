// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fuse_fc_swiglu_to_gated_mlp.hpp"

#include "intel_gpu/op/gated_mlp.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

#include <numeric>

namespace ov::intel_gpu {

FuseFCSwiGLUToGatedMLP::FuseFCSwiGLUToGatedMLP() {
    using namespace ov::pass::pattern;

    auto src = any_input();
    auto w_up = any_input();
    auto w_gate = any_input();
    auto w_down = any_input();

    auto mm_up = wrap_type<ov::op::v0::MatMul>({src, w_up});
    auto mm_gate = wrap_type<ov::op::v0::MatMul>({src, w_gate});
    auto swish = wrap_type<ov::op::v4::Swish>({mm_gate});
    auto mul_0 = wrap_type<ov::op::v1::Multiply>({swish, mm_up});
    auto mul_1 = wrap_type<ov::op::v1::Multiply>({mm_up, swish});
    auto mul = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{mul_0, mul_1});
    auto mm_down = wrap_type<ov::op::v0::MatMul>({mul, w_down});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pm = m.get_pattern_value_map();
        auto down = ov::as_type_ptr<ov::op::v0::MatMul>(pm.at(mm_down).get_node_shared_ptr());
        auto up = ov::as_type_ptr<ov::op::v0::MatMul>(pm.at(mm_up).get_node_shared_ptr());
        auto gate = ov::as_type_ptr<ov::op::v0::MatMul>(pm.at(mm_gate).get_node_shared_ptr());
        auto sw = ov::as_type_ptr<ov::op::v4::Swish>(pm.at(swish).get_node_shared_ptr());
        ov::NodeVector new_ops;

        if (!down || !up || !gate || !sw || transformation_callback(down))
            return false;

        if (down->get_transpose_a() || up->get_transpose_a() || gate->get_transpose_a())
            return false;

        if (up->input_value(0) != gate->input_value(0))
            return false;

        // Swish can be represented as 1-input(default beta=1.0) or 2-input(beta tensor).
        if (sw->get_input_size() != 1 && sw->get_input_size() != 2)
            return false;

        if (sw->get_input_size() == 2) {
            auto beta_const = ov::as_type_ptr<ov::op::v0::Constant>(sw->get_input_node_shared_ptr(1));
            if (!beta_const)
                return false;
            auto beta = beta_const->cast_vector<float>();
            if (beta.empty() || std::fabs(beta[0] - 1.0f) > 1e-6f)
                return false;
        }

        auto create_transpose = [&](const ov::Output<ov::Node>& node, const std::string& transpose_name) {
            std::vector<size_t> transpose_order(node.get_partial_shape().size());
            std::iota(transpose_order.begin(), transpose_order.end(), 0);
            std::swap(*(transpose_order.end() - 1), *(transpose_order.end() - 2));

            auto transpose_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{transpose_order.size()}, transpose_order);
            auto transpose = std::make_shared<ov::op::v1::Transpose>(node, transpose_const);
            if (!ov::is_type<ov::op::v0::Constant>(transpose)) {
                new_ops.push_back(transpose_const);
                MatcherPass::register_new_node(transpose);
            }
            transpose->set_friendly_name(transpose_name);
            ov::disable_constant_folding(transpose);
            new_ops.push_back(transpose);
            return transpose;
        };

        auto gate_weights = gate->get_transpose_b()
                                ? create_transpose(gate->input_value(1), gate->get_friendly_name() + "/transpose_b_for_gmlp")
                                : gate->input_value(1);
        auto up_weights = up->get_transpose_b()
                              ? create_transpose(up->input_value(1), up->get_friendly_name() + "/transpose_b_for_gmlp")
                              : up->input_value(1);
        auto down_weights = down->get_transpose_b()
                                ? create_transpose(down->input_value(1), down->get_friendly_name() + "/transpose_b_for_gmlp")
                                : down->input_value(1);

        auto gmlp = std::make_shared<ov::intel_gpu::op::GatedMLP>(
            up->input_value(0),
            gate_weights,
            up_weights,
            down_weights,
            ov::op::internal::GLU::GluType::Swish,
            down->get_output_element_type(0));

        gmlp->set_friendly_name(down->get_friendly_name());
        new_ops.push_back(gmlp);
        ov::copy_runtime_info(m.get_matched_nodes(), new_ops);
        ov::replace_node(down, gmlp);
        return true;
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(mm_down, "FuseFCSwiGLUToGatedMLP");
    register_matcher(matcher, callback);
}

}  // namespace ov::intel_gpu
