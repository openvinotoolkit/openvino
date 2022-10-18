// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/clamp_fusion.hpp"

#include <memory>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <vector>

#include "itt.hpp"
#include "transformations/utils/utils.hpp"

ngraph::pass::ClampFusion::ClampFusion() {
    MATCHER_SCOPE(ClampFusion);
    auto data_pattern = ngraph::pattern::any_input();
    auto min_const_pattern = ngraph::pattern::wrap_type<opset5::Constant>();
    auto max_const_pattern = ngraph::pattern::wrap_type<opset5::Constant>();
    auto max_pattern1 =
        ngraph::pattern::wrap_type<opset5::Maximum>({data_pattern, min_const_pattern}, pattern::consumers_count(1));
    auto min_pattern1 = ngraph::pattern::wrap_type<opset5::Minimum>({max_pattern1, max_const_pattern});
    auto min_pattern2 = ngraph::pattern::wrap_type<opset5::Minimum>({data_pattern, max_const_pattern});
    auto max_pattern2 =
        ngraph::pattern::wrap_type<opset5::Maximum>({min_pattern2, min_const_pattern}, pattern::consumers_count(1));
    auto root = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{min_pattern1, max_pattern2});

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto pattern_map = m.get_pattern_value_map();
        auto data = pattern_map.at(data_pattern);
        auto min_const =
            std::dynamic_pointer_cast<opset5::Constant>(pattern_map.at(min_const_pattern).get_node_shared_ptr());
        if (!min_const)
            return false;
        if (shape_size(min_const->get_shape()) != 1)
            return false;
        auto max_const =
            std::dynamic_pointer_cast<opset5::Constant>(pattern_map.at(max_const_pattern).get_node_shared_ptr());
        if (!max_const)
            return false;
        if (shape_size(max_const->get_shape()) != 1)
            return false;

        double min_value = min_const->cast_vector<double>()[0];
        double max_value = max_const->cast_vector<double>()[0];

        auto clamp = register_new_node<ngraph::opset5::Clamp>(data, min_value, max_value);

        std::shared_ptr<ov::Node> root_node;
        NodeVector nodes;
        auto min_pattern1_it = pattern_map.find(min_pattern1);
        if (min_pattern1_it != std::end(pattern_map)) {
            root_node = min_pattern1_it->second.get_node_shared_ptr();
            nodes.push_back(pattern_map.at(max_pattern1).get_node_shared_ptr());
        } else {
            root_node = pattern_map.at(max_pattern2).get_node_shared_ptr();
            nodes.push_back(pattern_map.at(min_pattern2).get_node_shared_ptr());
        }
        nodes.push_back(root_node);

        clamp->set_friendly_name(root_node->get_friendly_name());

        copy_runtime_info(nodes, clamp);
        replace_node(root_node, clamp);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(root, matcher_name);
    this->register_matcher(m, callback);
}
