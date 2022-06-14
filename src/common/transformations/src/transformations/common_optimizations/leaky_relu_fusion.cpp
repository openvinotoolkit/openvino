// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/leaky_relu_fusion.hpp"

#include <memory>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <vector>

#include "itt.hpp"
#include "transformations/utils/utils.hpp"

ngraph::pass::LeakyReluFusion::LeakyReluFusion() {
    MATCHER_SCOPE(LeakyReluFusion);
    auto data_pattern = ngraph::pattern::any_input();
    auto alpha_pattern = ngraph::pattern::any_input(pattern::has_static_shape());
    auto multiply_pattern =
        ngraph::pattern::wrap_type<opset8::Multiply>({data_pattern, alpha_pattern}, pattern::consumers_count(1));
    auto max_pattern = ngraph::pattern::wrap_type<opset8::Maximum>({data_pattern, multiply_pattern});

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto& original_alpha_pattern = pattern_map.at(alpha_pattern);

        if (shape_size(original_alpha_pattern.get_shape()) != 1)
            return false;

        auto leaky_relu =
            register_new_node<ngraph::opset8::PRelu>(pattern_map.at(data_pattern), original_alpha_pattern);
        auto maximum = pattern_map.at(max_pattern);
        leaky_relu->set_friendly_name(maximum.get_node()->get_friendly_name());

        copy_runtime_info({pattern_map.at(multiply_pattern).get_node_shared_ptr(), maximum.get_node_shared_ptr()},
                          leaky_relu);
        replace_node(maximum.get_node_shared_ptr(), leaky_relu);
        MATCHER_SCOPE_ENABLE(LeakyReluFusion);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(max_pattern, matcher_name);
    this->register_matcher(m, callback);
}
