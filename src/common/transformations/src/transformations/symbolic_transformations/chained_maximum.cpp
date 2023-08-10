// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include "transformations/symbolic_transformations/chained_maximum.hpp"
#include "openvino/core/dimension_tracker.hpp"
#include "transformations/symbolic_transformations/utils.hpp"

#include <openvino/op/maximum.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>

ov::pass::ChainedMaximumOptimization::ChainedMaximumOptimization() {
    MATCHER_SCOPE(ChainedMaximumOptimization);
    auto first_input = pattern::any_input();
    auto second_input = pattern::any_input();
    auto third_input = pattern::any_input();
    auto upper_maximum = pattern::wrap_type<op::v1::Maximum>({first_input, second_input});
    auto lower_maximum = pattern::wrap_type<op::v1::Maximum>({upper_maximum, third_input});

    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
        const auto& vm = m.get_pattern_value_map();
        auto get_labels = [&](const std::shared_ptr<Node>& node_label) {
            return vm.at(node_label).get_tensor().get_value_label();
        };

        auto output_to_replace = vm.at(upper_maximum);
        if (are_unique_and_equal_labels(get_labels(first_input), get_labels(third_input))) {
            // optimized graph is Maximum(second_input, third_input)
            return ov::replace_output_update_name(output_to_replace, vm.at(second_input));
        } else if (are_unique_and_equal_labels(get_labels(second_input), get_labels(third_input))) {
            // optimized graph is Maximum(first_input, third_input)
            return ov::replace_output_update_name(output_to_replace, vm.at(first_input));
        }
        return false;
    };

    auto m = std::make_shared<pattern::Matcher>(lower_maximum, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
