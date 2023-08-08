// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/symbolic_transformations/chained_maximum.hpp"

#include <openvino/op/maximum.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>

#include "itt.hpp"

ov::pass::ChainedMaximumOptimization::ChainedMaximumOptimization() {
    MATCHER_SCOPE(ChainedMaximumOptimization);
    auto first_input = pattern::any_input();
    auto second_input = pattern::any_input();
    auto third_input = pattern::any_input();
    auto upper_max_label = pattern::wrap_type<op::v1::Maximum>({first_input, second_input});
    auto lower_max_label = pattern::wrap_type<op::v1::Maximum>({upper_max_label, third_input});

    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto first_labels = pattern_to_output.at(first_input).get_tensor().get_value_label();
        auto second_labels = pattern_to_output.at(second_input).get_tensor().get_value_label();
        auto third_labels = pattern_to_output.at(third_input).get_tensor().get_value_label();

        auto valid_labels = [](const ov::TensorLabel& labels) {
            return !labels.empty() && std::all_of(labels.begin(), labels.end(), [](const label_t& l) {
                return l != 0;
            });
        };
        bool replaced = false;
        auto intermidiate = pattern_to_output.at(upper_max_label);
        if (valid_labels(first_labels) && valid_labels(third_labels) && first_labels == third_labels) {
            // Maximum(second_input, third_input)
            intermidiate.replace(pattern_to_output.at(second_input));
            replaced = true;
        } else if (valid_labels(second_labels) && valid_labels(third_labels) && second_labels == third_labels) {
            // Maximum(first_input, third_input)
            intermidiate.replace(pattern_to_output.at(first_input));
            replaced = true;
        }
        return replaced;
    };

    auto m = std::make_shared<pattern::Matcher>(lower_max_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
