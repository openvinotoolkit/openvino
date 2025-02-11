// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/softplus_fusion.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/log.hpp"
#include "openvino/op/softplus.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

ov::pass::SoftPlusFusion::SoftPlusFusion() {
    MATCHER_SCOPE(SoftPlusFusion);
    // fuses ln(exp(x) + 1.0) operations into SoftPlus(x)
    auto input = pattern::any_input();
    auto exp = std::make_shared<ov::op::v0::Exp>(input);
    auto add_constant =
        ov::pass::pattern::wrap_type<ov::op::v0::Constant>(pattern::type_matches_any({element::f32, element::f16}));
    auto add = std::make_shared<ov::op::v1::Add>(exp, add_constant);
    auto log = std::make_shared<ov::op::v0::Log>(add);

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto exp_input = pattern_to_output.at(input);

        auto constant = ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(add_constant).get_node_shared_ptr());
        if (!constant)
            return false;

        auto data = constant->cast_vector<float>();
        if (data.size() != 1 || data[0] != 1.0) {
            return false;
        }

        auto softplus = std::make_shared<ov::op::v4::SoftPlus>(exp_input);

        softplus->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info({pattern_to_output.at(log).get_node_shared_ptr(),
                               pattern_to_output.at(add).get_node_shared_ptr(),
                               pattern_to_output.at(exp).get_node_shared_ptr()},
                              softplus);
        ov::replace_node(m.get_match_root(), softplus);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(log, matcher_name);
    register_matcher(m, callback);
}
