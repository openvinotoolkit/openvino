// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/clamp_fusion.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::ClampFusion::ClampFusion() {
    MATCHER_SCOPE(ClampFusion);
    auto data_pattern = pass::pattern::any_input();
    auto min_const_pattern = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto max_const_pattern = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto max_pattern1 = ov::pass::pattern::wrap_type<ov::op::v1::Maximum>({data_pattern, min_const_pattern},
                                                                          pattern::consumers_count(1));
    auto min_pattern1 = ov::pass::pattern::wrap_type<ov::op::v1::Minimum>({max_pattern1, max_const_pattern});
    auto min_pattern2 = ov::pass::pattern::wrap_type<ov::op::v1::Minimum>({data_pattern, max_const_pattern});
    auto max_pattern2 = ov::pass::pattern::wrap_type<ov::op::v1::Maximum>({min_pattern2, min_const_pattern},
                                                                          pattern::consumers_count(1));
    auto root = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{min_pattern1, max_pattern2});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        auto pattern_map = m.get_pattern_value_map();
        auto data = pattern_map.at(data_pattern);
        auto min_const = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(min_const_pattern).get_node_shared_ptr());
        if (!min_const)
            return false;
        if (shape_size(min_const->get_shape()) != 1)
            return false;
        auto max_const = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(max_const_pattern).get_node_shared_ptr());
        if (!max_const)
            return false;
        if (shape_size(max_const->get_shape()) != 1)
            return false;

        double min_value = min_const->cast_vector<double>()[0];
        double max_value = max_const->cast_vector<double>()[0];
        if (min_value > max_value)
            return false;

        auto clamp = register_new_node<ov::op::v0::Clamp>(data, min_value, max_value);

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

    auto m = std::make_shared<ov::pass::pattern::Matcher>(root, matcher_name);
    this->register_matcher(m, callback);
}
