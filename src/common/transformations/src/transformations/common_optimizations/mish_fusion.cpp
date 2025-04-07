// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/mish_fusion.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/log.hpp"
#include "openvino/op/mish.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

ov::pass::MishFusion::MishFusion() {
    MATCHER_SCOPE(MishFusion);
    auto input = pass::pattern::any_input();
    auto exp = std::make_shared<ov::op::v0::Exp>(input);
    auto add = std::make_shared<ov::op::v1::Add>(exp, ov::pass::pattern::wrap_type<ov::op::v0::Constant>());
    auto log = std::make_shared<ov::op::v0::Log>(add);
    auto tanh = std::make_shared<ov::op::v0::Tanh>(log);
    auto mul = std::make_shared<ov::op::v1::Multiply>(input, tanh);

    ov::matcher_pass_callback matcher_pass_callback = [=](ov::pass::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto exp_input = pattern_to_output.at(input);

        auto mish = std::make_shared<ov::op::v4::Mish>(exp_input);

        mish->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info({pattern_to_output.at(mul).get_node_shared_ptr(),
                               pattern_to_output.at(tanh).get_node_shared_ptr(),
                               pattern_to_output.at(log).get_node_shared_ptr(),
                               pattern_to_output.at(add).get_node_shared_ptr(),
                               pattern_to_output.at(exp).get_node_shared_ptr()},
                              mish);
        ov::replace_node(m.get_match_root(), mish);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(mul, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
