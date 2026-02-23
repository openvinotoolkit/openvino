// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/leaky_relu_fusion.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/prelu.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;

namespace ov::pass {

LeakyReluFusion::LeakyReluFusion() {
    MATCHER_SCOPE(LeakyReluFusion);
    auto data_pattern = pattern::any_input();
    auto alpha_pattern = pattern::wrap_type<v0::Constant>();
    auto multiply_pattern =
        pattern::wrap_type<v1::Multiply>({data_pattern, alpha_pattern}, pattern::consumers_count(1));
    auto max_pattern = pattern::wrap_type<v1::Maximum>({data_pattern, multiply_pattern});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto& original_alpha_pattern = pattern_map.at(alpha_pattern);

        if (shape_size(original_alpha_pattern.get_shape()) != 1)
            return false;

        auto constant = ov::as_type_ptr<v0::Constant>(original_alpha_pattern.get_node_shared_ptr());
        if (!constant)
            return false;

        float value;
        if (!ov::op::util::get_single_value(constant, value))
            return false;

        if (value > 1.0f)
            return false;

        auto leaky_relu = register_new_node<v0::PRelu>(pattern_map.at(data_pattern), original_alpha_pattern);
        auto maximum = pattern_map.at(max_pattern);
        leaky_relu->set_friendly_name(maximum.get_node()->get_friendly_name());

        copy_runtime_info({pattern_map.at(multiply_pattern).get_node_shared_ptr(), maximum.get_node_shared_ptr()},
                          leaky_relu);
        replace_node(maximum.get_node_shared_ptr(), leaky_relu);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(max_pattern, matcher_name);
    this->register_matcher(m, callback);
}

}  // namespace ov::pass
