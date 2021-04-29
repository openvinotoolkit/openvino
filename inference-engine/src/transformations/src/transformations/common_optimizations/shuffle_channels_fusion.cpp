// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/shuffle_channels_fusion.hpp"
#include "itt.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset6.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

bool check_shapes(const ngraph::Shape& shape_input, const ngraph::Shape& shape_reshape_before,
                  const ngraph::AxisVector& transpose_constant_values, const ngraph::Shape& shape_reshape_after) {
    // x: [N, C, H, W]
    bool is_transformation_valid = (shape_input.size() == 4);

    // x'= reshape(x, [N, group, C / group, H * W]) or reshape(x, [N, group, C / group, H, W])
    bool is_reshape_before_valid = (shape_reshape_before.size() == 4 || shape_reshape_before.size() == 5);
    if (is_reshape_before_valid) {
        size_t group = shape_reshape_before[1];
        ngraph::Shape expected_reshape_before = { shape_input[0], group, shape_input[1] / group };

        if (shape_reshape_before.size() == 4) {
            expected_reshape_before.push_back(shape_input[2] * shape_input[3]);
        } else {
            expected_reshape_before.push_back(shape_input[2]);
            expected_reshape_before.push_back(shape_input[3]);
        }

        is_reshape_before_valid &= (expected_reshape_before == shape_reshape_before);
    }

    // x''= transpose(x', [0, 2, 1, 3]) or transpose(x', [0, 2, 1, 3, 4])
    bool is_transpose_valid = (transpose_constant_values.size() == 4 || transpose_constant_values.size() == 5);
    if (is_transpose_valid) {
        ngraph::AxisVector expected_transpose_values{ 0, 2, 1, 3 };
        if (transpose_constant_values.size() == 5) {
            expected_transpose_values.push_back(4);
        }

        is_transpose_valid &= (expected_transpose_values == transpose_constant_values);
    }

    // y = reshape(x'', [N, C, H, W])
    bool is_reshape_after_valid = (shape_input == shape_reshape_after);

    is_transformation_valid &= is_reshape_before_valid & is_transpose_valid & is_reshape_after_valid;
    return is_transformation_valid;
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::ShuffleChannelsFusion, "ShuffleChannelsFusion", 0);

ngraph::pass::ShuffleChannelsFusion::ShuffleChannelsFusion(const bool reshape_constants_check) {
    MATCHER_SCOPE(ShuffleChannelsFusion);
    auto has_static_4d_shape = [](const Output<Node>& output) {
        return pattern::has_static_shape()(output) && pattern::rank_equals(4)(output);
    };

    auto input = ngraph::pattern::any_input(has_static_4d_shape);
    auto reshape_before_const_pattern = ngraph::pattern::wrap_type<ngraph::opset6::Constant>();
    auto transpose_const_pattern = ngraph::pattern::wrap_type<ngraph::opset6::Constant>();
    auto reshape_after_const_pattern = ngraph::pattern::wrap_type<ngraph::opset6::Constant>();

    auto has_static_shape_and_single_consumer = [](const Output<Node>& output) {
        return pattern::has_static_shape()(output) && pattern::consumers_count(1)(output);
    };
    auto reshape_before_pattern = ngraph::pattern::wrap_type<ngraph::opset6::Reshape>({input, reshape_before_const_pattern},
                                                                                      has_static_shape_and_single_consumer);
    auto transpose_pattern = ngraph::pattern::wrap_type<ngraph::opset6::Transpose>({reshape_before_pattern, transpose_const_pattern},
                                                                                   has_static_shape_and_single_consumer);
    auto reshape_after_pattern = ngraph::pattern::wrap_type<ngraph::opset6::Reshape>({transpose_pattern, reshape_after_const_pattern},
                                                                                     pattern::has_static_shape());

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto data = pattern_map.at(input);
        auto reshape_before = std::dynamic_pointer_cast<ngraph::opset6::Reshape>(pattern_map.at(reshape_before_pattern).get_node_shared_ptr());
        auto transpose = std::dynamic_pointer_cast<ngraph::opset6::Transpose>(pattern_map.at(transpose_pattern).get_node_shared_ptr());
        auto reshape_after = std::dynamic_pointer_cast<ngraph::opset6::Reshape>(pattern_map.at(reshape_after_pattern).get_node_shared_ptr());
        if (!reshape_after || !transpose || !reshape_after)
            return false;

        if (reshape_constants_check) {
            auto reshape_before_constant = std::dynamic_pointer_cast<ngraph::opset6::Constant>(
                pattern_map.at(reshape_before_const_pattern).get_node_shared_ptr());
            auto reshape_after_constant = std::dynamic_pointer_cast<ngraph::opset6::Constant>(
                pattern_map.at(reshape_after_const_pattern).get_node_shared_ptr());

            if (!reshape_before_constant || !reshape_after_constant)
                return false;

            const auto& reshape_before_values = reshape_before_constant->cast_vector<int64_t>();
            const auto& reshape_after_values = reshape_after_constant->cast_vector<int64_t>();
            if (std::any_of(reshape_before_values.cbegin(), reshape_before_values.cend(), [](const int64_t& value) { return value == -1; }) ||
                std::any_of(reshape_after_values.cbegin(), reshape_after_values.cend(), [](const int64_t& value) { return value == -1; })) {
                return false;
            }
        }

        auto shape_input = reshape_before->get_input_shape(0);
        auto shape_reshape_before = reshape_before->get_output_shape(0);
        auto shape_reshape_after = reshape_after->get_output_shape(0);

        auto transpose_constant = std::dynamic_pointer_cast<ngraph::opset6::Constant>(pattern_map.at(transpose_const_pattern).get_node_shared_ptr());
        auto transpose_constant_values = transpose_constant->get_axis_vector_val();
        if (!check_shapes(shape_input, shape_reshape_before, transpose_constant_values, shape_reshape_after))
            return false;

        int64_t axis = 1ul;
        int64_t group = shape_reshape_before[1];

        auto shuffle_shannels = std::make_shared<ngraph::opset6::ShuffleChannels>(data, axis, group);
        shuffle_shannels->set_friendly_name(reshape_after->get_friendly_name());
        ngraph::copy_runtime_info({ reshape_before, transpose, reshape_after }, shuffle_shannels);
        ngraph::replace_node(reshape_after, shuffle_shannels);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(reshape_after_pattern, matcher_name);
    register_matcher(m, callback);
}
