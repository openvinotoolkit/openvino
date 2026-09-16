// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fuse_clamp_and_fake_quantize.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace v0 = ov::op::v0;

namespace ov::pass {

FuseClampAndFakeQuantize::FuseClampAndFakeQuantize() {
    MATCHER_SCOPE(FuseClampAndFakeQuantize);

    auto data_pattern = pattern::any_input();
    auto clamp_pattern = pattern::wrap_type<v0::Clamp>({data_pattern}, pattern::consumers_count(1));
    auto input_low_pattern = pattern::wrap_type<v0::Constant>();
    auto input_high_pattern = pattern::wrap_type<v0::Constant>();
    auto fq_pattern = pattern::wrap_type<v0::FakeQuantize>(
        {clamp_pattern, input_low_pattern, input_high_pattern, pattern::any_input(), pattern::any_input()});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto pattern_map = m.get_pattern_value_map();
        const auto clamp = ov::as_type_ptr<v0::Clamp>(pattern_map.at(clamp_pattern).get_node_shared_ptr());
        const auto input_low = ov::as_type_ptr<v0::Constant>(pattern_map.at(input_low_pattern).get_node_shared_ptr());
        const auto input_high = ov::as_type_ptr<v0::Constant>(pattern_map.at(input_high_pattern).get_node_shared_ptr());
        if (!clamp || !input_low || !input_high) {
            return false;
        }

        const auto input_low_values = input_low->cast_vector<float>();
        const auto input_high_values = input_high->cast_vector<float>();
        const auto clamp_low = static_cast<float>(clamp->get_min());
        const auto clamp_high = static_cast<float>(clamp->get_max());

        if (!std::all_of(input_low_values.begin(), input_low_values.end(), [&](auto v) {
                return v >= clamp_low;
            })) {
            return false;
        }

        if (!std::all_of(input_high_values.begin(), input_high_values.end(), [&](auto v) {
                return v <= clamp_high;
            })) {
            return false;
        }

        return ov::replace_output_update_name(clamp->output(0), clamp->input_value(0));
    };

    auto m = std::make_shared<pattern::Matcher>(fq_pattern, matcher_name);
    register_matcher(m, callback);
}

}  // namespace ov::pass