// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/clamp_fusion.hpp"
#include "transformations/utils/utils.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset5.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "itt.hpp"


NGRAPH_RTTI_DEFINITION(ngraph::pass::ClampFusion, "ClampFusion", 0);

ngraph::pass::ClampFusion::ClampFusion() {
    MATCHER_SCOPE(ClampFusion);
    auto data_pattern = ngraph::pattern::any_input();
    auto min_const_pattern = ngraph::pattern::wrap_type<opset5::Constant>();
    auto max_const_pattern = ngraph::pattern::wrap_type<opset5::Constant>();
    auto max_pattern = ngraph::pattern::wrap_type<opset5::Maximum>({data_pattern, min_const_pattern}, pattern::consumers_count(1));
    auto min_pattern = ngraph::pattern::wrap_type<opset5::Minimum>({max_pattern, max_const_pattern});

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto pattern_map = m.get_pattern_value_map();
        auto data = pattern_map.at(data_pattern);
        auto min_const = std::dynamic_pointer_cast<opset5::Constant>(pattern_map.at(min_const_pattern).get_node_shared_ptr());
        if (!min_const)
            return false;
        if (shape_size(min_const->get_shape()) != 1)
            return false;
        auto max_const = std::dynamic_pointer_cast<opset5::Constant>(pattern_map.at(max_const_pattern).get_node_shared_ptr());
        if (!max_const)
            return false;
        if (shape_size(max_const->get_shape()) != 1)
            return false;

        double min_value = min_const->cast_vector<double>()[0];
        double max_value = max_const->cast_vector<double>()[0];

        auto clamp = register_new_node<ngraph::opset5::Clamp>(data, min_value, max_value);
        auto minimum = pattern_map.at(min_pattern);
        clamp->set_friendly_name(minimum.get_node()->get_friendly_name());

        copy_runtime_info({
                            pattern_map.at(max_pattern).get_node_shared_ptr(),
                            minimum.get_node_shared_ptr()
                          },
                          clamp);
        replace_node(minimum.get_node_shared_ptr(), clamp);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(min_pattern, matcher_name);
    this->register_matcher(m, callback);
}
