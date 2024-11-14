// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/adaptive_pool_to_reduce.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/adaptive_avg_pool.hpp"
#include "openvino/op/adaptive_max_pool.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::op;

ov::pass::AdaptivePoolToReduce::AdaptivePoolToReduce() {
    MATCHER_SCOPE(AdaptivePoolToReduce);
    auto data_pattern = pattern::any_input();
    auto out_spatial_shape = pattern::wrap_type<v0::Constant>();
    auto a_pool = pattern::wrap_type<v8::AdaptiveAvgPool, v8::AdaptiveMaxPool>({data_pattern, out_spatial_shape});

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_map();

        const auto& spatial_shape_c = ov::as_type_ptr<v0::Constant>(pattern_map.at(out_spatial_shape));
        auto spatial_shape = spatial_shape_c->cast_vector<int64_t>();
        // Verify that all dimensions in adaptive pool shape are 1
        for (auto& s : spatial_shape) {
            if (s != 1)
                return false;
        }

        auto axes = std::vector<int64_t>(spatial_shape.size(), 0);
        std::iota(axes.begin(), axes.end(), 2);
        auto axes_const = v0::Constant::create(element::i64, {spatial_shape.size()}, axes);
        const auto adaptive_pool = pattern_map.at(a_pool);
        std::shared_ptr<Node> res_node;
        if (ov::as_type_ptr<v8::AdaptiveAvgPool>(adaptive_pool)) {
            res_node = std::make_shared<v1::ReduceMean>(adaptive_pool->input_value(0), axes_const, true);
        } else if (ov::as_type_ptr<v8::AdaptiveMaxPool>(adaptive_pool)) {
            if (adaptive_pool->outputs().size() > 1 && adaptive_pool->output(1).get_target_inputs().size() != 0) {
                // If indexes are used we can't replace it
                return false;
            }
            res_node = std::make_shared<v1::ReduceMax>(adaptive_pool->input_value(0), axes_const, true);
        } else {
            return false;
        }
        adaptive_pool->output(0).replace(res_node);
        res_node->set_friendly_name(adaptive_pool->get_friendly_name());
        copy_runtime_info(adaptive_pool, res_node);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(a_pool, matcher_name);
    this->register_matcher(m, callback);
}
