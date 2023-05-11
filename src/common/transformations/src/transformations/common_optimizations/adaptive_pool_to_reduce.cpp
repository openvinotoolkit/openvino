// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/adaptive_pool_to_reduce.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/opsets/opset11.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::AdaptivePoolToReduce::AdaptivePoolToReduce() {
    MATCHER_SCOPE(AdaptivePoolToReduce);
    auto data_pattern = pattern::any_input();
    auto out_spatial_shape = pattern::wrap_type<opset11::Constant>();
    auto a_avg_pool = pattern::wrap_type<opset11::AdaptiveAvgPool>({data_pattern, out_spatial_shape});
    auto a_max_pool = pattern::wrap_type<opset11::AdaptiveMaxPool>({data_pattern, out_spatial_shape});
    auto adaptive_pool = std::make_shared<pattern::op::Or>(OutputVector{a_avg_pool, a_max_pool});

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        const auto& spatial_shape_c =
            std::dynamic_pointer_cast<opset11::Constant>(pattern_map.at(out_spatial_shape).get_node_shared_ptr());
        auto spatial_shape = spatial_shape_c->cast_vector<int64_t>();
        // Verify that all dimensions in adaptive pool shape are 1
        for (auto& s : spatial_shape) {
            if (s != 1)
                return false;
        }

        auto axes = std::vector<int64_t>(spatial_shape.size(), 0);
        std::iota(axes.begin(), axes.end(), 2);
        auto axes_const = opset11::Constant::create(element::i64, {spatial_shape.size()}, axes);
        if (pattern_map.count(a_avg_pool)) {
            const auto adaptive_avg_pool = pattern_map.at(a_avg_pool).get_node_shared_ptr();
            auto res_node = std::make_shared<opset11::ReduceMean>(adaptive_avg_pool->input_value(0), axes_const, true);
            replace_node(adaptive_avg_pool, res_node);
            res_node->set_friendly_name(adaptive_avg_pool->get_friendly_name());
            copy_runtime_info(adaptive_avg_pool, res_node);
        } else if (pattern_map.count(a_max_pool)) {
            const auto adaptive_max_pool = pattern_map.at(a_max_pool).get_node_shared_ptr();
            auto res_node = std::make_shared<opset11::ReduceMax>(adaptive_max_pool->input_value(0), axes_const, true);
            replace_node(adaptive_max_pool, res_node);
            res_node->set_friendly_name(adaptive_max_pool->get_friendly_name());
            copy_runtime_info(adaptive_max_pool, res_node);
        } else {
            return false;
        }

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(adaptive_pool, matcher_name);
    this->register_matcher(m, callback);
}
