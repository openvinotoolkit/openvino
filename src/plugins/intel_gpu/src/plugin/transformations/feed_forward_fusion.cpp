// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "feed_forward_fusion.hpp"

#include "intel_gpu/op/feed_forward.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace intel_gpu {

FeedForwardFusion::FeedForwardFusion() {
    using namespace ov::pass::pattern;

    auto matmul1 = any_input();
    auto bias = any_input();
    auto add1 = wrap_type<ov::op::v1::Add>({matmul1, bias}, consumers_count(5));
    auto mul1 = wrap_type<ov::op::v1::Multiply>({add1, add1});
    auto mul2 = wrap_type<ov::op::v1::Multiply>({add1, mul1});
    auto mul3_const = any_input();
    auto mul3 = wrap_type<ov::op::v1::Multiply>({mul2, mul3_const});
    auto add2 = wrap_type<ov::op::v1::Add>({add1, mul3});
    auto mul4_const = any_input();
    auto mul4 = wrap_type<ov::op::v1::Multiply>({add2, mul4_const});
    auto tanh = wrap_type<ov::op::v0::Tanh>({mul4});
    auto add3_const = any_input();
    auto add3 = wrap_type<ov::op::v1::Add>({tanh, add3_const});
    auto mul5 = wrap_type<ov::op::v1::Multiply>({add3, add1});
    auto mul6_const = any_input();
    auto mul6 = wrap_type<ov::op::v1::Multiply>({mul5, mul6_const});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto& m_add1 = pattern_map.at(add1).get_node_shared_ptr();
        const auto& m_mul3_const = pattern_map.at(mul3_const).get_node_shared_ptr();
        const auto& m_mul4_const = pattern_map.at(mul4_const).get_node_shared_ptr();
        const auto& m_add3_const = pattern_map.at(add3_const).get_node_shared_ptr();
        const auto& m_mul6_const = pattern_map.at(mul6_const).get_node_shared_ptr();

        auto in0_ss = ov::shape_size(m_add1->get_shape());
        auto in1_ss = ov::shape_size(m_mul3_const->get_shape());
        auto in2_ss = ov::shape_size(m_mul4_const->get_shape());
        auto in3_ss = ov::shape_size(m_add3_const->get_shape());
        auto in4_ss = ov::shape_size(m_mul6_const->get_shape());

        auto same_shape = (in0_ss == in1_ss) && (in1_ss == in2_ss) && (in2_ss == in3_ss) && (in3_ss == in4_ss);
        auto is_scalar = (in1_ss == 1) && (in2_ss == 1) && (in3_ss == 1) && ((in4_ss == 1));

        if (!same_shape && !is_scalar)
            return false;

        auto ff = std::make_shared<op::FeedForward>(m_add1, m_mul3_const, m_mul4_const, m_add3_const, m_mul6_const);
        ff->set_friendly_name(m.get_match_root()->get_friendly_name() + "_ff");
        ov::copy_runtime_info(m.get_matched_nodes(), ff);
        ov::replace_node(m.get_match_root(), ff);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(mul6, "FeedForwardFusion");
    this->register_matcher(m, callback);
}

}  // namespace intel_gpu
}  // namespace ov
