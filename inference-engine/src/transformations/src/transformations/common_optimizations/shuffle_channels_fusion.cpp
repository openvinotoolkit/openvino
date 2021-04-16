// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/shuffle_channels_fusion.hpp"
#include "transformations/utils/utils.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset7.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "itt.hpp"


NGRAPH_RTTI_DEFINITION(ngraph::pass::ShuffleChannelsFusion, "ShuffleChannelsFusion", 0);

ngraph::pass::ShuffleChannelsFusion::ShuffleChannelsFusion() {
    MATCHER_SCOPE(ShuffleChannelsFusion);
    auto data_pattern = ngraph::pattern::any_input(pattern::has_static_shape());
    auto shape1_pattern = ngraph::pattern::wrap_type<opset7::Constant>();
    auto reshape1_pattern = ngraph::pattern::wrap_type<opset7::Reshape>({data_pattern, shape1_pattern}, pattern::consumers_count(1));
    auto perm_pattern = ngraph::pattern::wrap_type<opset7::Constant>();
    auto transpose_pattern = ngraph::pattern::wrap_type<opset7::Transpose>({reshape1_pattern, perm_pattern});
    auto shape2_pattern = ngraph::pattern::wrap_type<opset7::Constant>();
    auto reshape2_pattern = ngraph::pattern::wrap_type<opset7::Reshape>({transpose_pattern, shape2_pattern});

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto& data = pattern_map.at(data_pattern);

        const auto& perm = pattern_map.at(perm_pattern).get_node_shared_ptr();
        auto perm_const = std::dynamic_pointer_cast<opset7::Constant>(perm);
        if (!perm_const)
            return false;
        auto perm_value = perm_const->cast_vector<int64_t>();
        if (perm_value.size() != 5 ||
            !(perm_value[0] == 0 &&
              perm_value[1] == 2 &&
              perm_value[2] == 1 &&
              perm_value[3] == 3 &&
              perm_value[4] == 4))
            return false;

        const auto& data_shape = data.get_shape();
        const auto& reshape1 = pattern_map.at(reshape1_pattern);
        const auto& reshape2 = pattern_map.at(reshape2_pattern);
        const auto& channel_splitting_shape = reshape1.get_shape();
        const auto& channel_concating_shape = reshape2.get_shape();

        if (data_shape.size() != 4 || channel_splitting_shape.size() != 5)
            return false;
        if (data_shape[0] != channel_splitting_shape[0] ||
            data_shape[1] != channel_splitting_shape[1] * channel_splitting_shape[2] ||
            data_shape[2] != channel_splitting_shape[3] ||
            data_shape[3] != channel_splitting_shape[4]) {
            return false;
        }
        if (data_shape != channel_concating_shape)
            return false;

        auto shuffle = register_new_node<ngraph::opset7::ShuffleChannels>(data, 1, channel_splitting_shape[1]);
        shuffle->set_friendly_name(reshape2.get_node()->get_friendly_name());

        copy_runtime_info({
                            reshape1.get_node_shared_ptr(),
                            pattern_map.at(transpose_pattern).get_node_shared_ptr(),
                            reshape2.get_node_shared_ptr(),
                          },
                          shuffle);
        replace_node(reshape2.get_node_shared_ptr(), shuffle);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(reshape2_pattern, matcher_name);
    this->register_matcher(m, callback);
}
