// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/mvn_fusion.hpp"
#include "transformations/utils/utils.hpp"


#include <memory>
#include <iostream>
#include <vector>

#include <ngraph/opsets/opset4.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>


ngraph::pass::MVNFusion::MVNFusion() {
    auto data_pattern = ngraph::pattern::any_input();
    auto axes_pattern = ngraph::pattern::wrap_type<opset4::Constant>();
    auto mean_pattern = ngraph::pattern::wrap_type<opset4::ReduceMean>({data_pattern, axes_pattern});
    auto numerator_pattern = ngraph::pattern::wrap_type<opset4::Subtract>({data_pattern, mean_pattern});
    auto mean_for_variance_pattern = ngraph::pattern::wrap_type<opset4::ReduceMean>({data_pattern, axes_pattern});
    auto diff_pattern = ngraph::pattern::wrap_type<opset4::Subtract>({data_pattern, mean_for_variance_pattern});
    auto sqr_pattern = ngraph::pattern::wrap_type<opset4::Power>({diff_pattern, ngraph::pattern::wrap_type<opset4::Constant>()});
    auto variance_pattern = ngraph::pattern::wrap_type<opset4::ReduceMean>({sqr_pattern, axes_pattern});
    auto stddev_pattern = ngraph::pattern::wrap_type<opset4::Power>({variance_pattern, ngraph::pattern::wrap_type<opset4::Constant>()});
    auto eps_pattern = ngraph::pattern::wrap_type<opset4::Constant>();
    auto add_pattern = ngraph::pattern::wrap_type<opset4::Add>({stddev_pattern, eps_pattern});
    auto reciprocal_pattern = ngraph::pattern::wrap_type<opset4::Power>({add_pattern, ngraph::pattern::wrap_type<opset4::Constant>()});
    auto mul_pattern = ngraph::pattern::wrap_type<opset4::Multiply>({numerator_pattern, reciprocal_pattern});

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto pattern_map = m.get_pattern_value_map();
        auto eps = std::dynamic_pointer_cast<opset4::Constant>(pattern_map[eps_pattern].get_node_shared_ptr());
        if (!eps)
            return false;
        float eps_value = 0;
        if (!op::util::get_single_value(eps, eps_value))
            return false;
        auto axes = std::dynamic_pointer_cast<opset4::Constant>(pattern_map[axes_pattern].get_node_shared_ptr());
        if (!axes)
            return false;
        const auto& axes_value = axes->cast_vector<int32_t>();
        AxisSet axis_set;
        for (auto axis : axes_value)
            axis_set.insert(axis);
        auto data = pattern_map[data_pattern];
        auto mul = pattern_map[mul_pattern].get_node_shared_ptr();
        auto mvn = std::make_shared<opset4::MVN>(data, axis_set, true, eps_value);
        mvn->set_friendly_name(mul->get_friendly_name());
        ngraph::copy_runtime_info({
                                    pattern_map[axes_pattern].get_node_shared_ptr(),
                                    pattern_map[mean_pattern].get_node_shared_ptr(),
                                    pattern_map[numerator_pattern].get_node_shared_ptr(),
                                    pattern_map[mean_for_variance_pattern].get_node_shared_ptr(),
                                    pattern_map[diff_pattern].get_node_shared_ptr(),
                                    pattern_map[sqr_pattern].get_node_shared_ptr(),
                                    pattern_map[variance_pattern].get_node_shared_ptr(),
                                    pattern_map[stddev_pattern].get_node_shared_ptr(),
                                    pattern_map[eps_pattern].get_node_shared_ptr(),
                                    pattern_map[reciprocal_pattern].get_node_shared_ptr(),
                                    pattern_map[mul_pattern].get_node_shared_ptr(),
                                  }, mvn);
        ngraph::replace_node(mul, mvn);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(mul_pattern, "MVNFusion");
    this->register_matcher(m, callback);
}
