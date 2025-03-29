// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/nonzero_horizontal_fusion.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::NonZeroHorizontalFusion::NonZeroHorizontalFusion() {
    MATCHER_SCOPE(NonZeroHorizontalFusion);
    auto input_m = pass::pattern::any_input(ov::pass::pattern::consumers_more_than(1));
    auto nonzero_m = pass::pattern::wrap_type<ov::op::v3::NonZero>({input_m});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto nonzero = ov::as_type_ptr<ov::op::v3::NonZero>(pattern_map.at(nonzero_m).get_node_shared_ptr());
        const auto out_prc = nonzero->get_output_type();

        bool status = false;
        auto replace_if_nodes_match = [&](const ov::Input<ov::Node>& in) {
            auto in_node = in.get_node();
            auto cur_nonzero = ov::as_type<ov::op::v3::NonZero>(in_node);
            if (in_node != nonzero.get() && cur_nonzero && cur_nonzero->get_output_type() == out_prc) {
                status |= ov::replace_output_update_name(cur_nonzero->output(0), nonzero->output(0));
            }
        };

        const auto consumers = pattern_map.at(input_m).get_target_inputs();
        std::for_each(consumers.begin(), consumers.end(), replace_if_nodes_match);
        return status;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(nonzero_m, matcher_name);
    register_matcher(m, callback);
}
