// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/shuffle_channels_fusion.hpp"
#include "itt.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset3.hpp>
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

ngraph::pass::ShuffleChannelsFusion::ShuffleChannelsFusion() {
    MATCHER_SCOPE(ShuffleChannelsFusion);
    auto input0 = std::make_shared<pattern::op::Label>(element::f32);
    auto input1 = std::make_shared<pattern::op::Label>(element::i64);
    auto input2 = std::make_shared<pattern::op::Label>(element::i64);
    auto input3 = std::make_shared<pattern::op::Label>(element::i64);
    auto reshape_before = std::make_shared<ngraph::opset3::Reshape>(input0, input1, false);
    auto transpose = std::make_shared<ngraph::opset3::Transpose>(reshape_before, input2);
    auto reshape_after = std::make_shared<ngraph::opset3::Reshape>(transpose, input3, false);

    ngraph::matcher_pass_callback callback = [this](pattern::Matcher& m) {
        auto reshape_after = std::dynamic_pointer_cast<ngraph::opset3::Reshape>(m.get_match_root());
        if (!reshape_after) {
            return false;
        }

        auto transpose = std::dynamic_pointer_cast<ngraph::opset3::Transpose>(reshape_after->input_value(0).get_node_shared_ptr());
        if (!transpose || transpose->get_output_target_inputs(0).size() != 1) {
            return false;
        }

        auto transpose_constant = std::dynamic_pointer_cast<ngraph::opset3::Constant>(transpose->input_value(1).get_node_shared_ptr());
        if (!transpose_constant) {
            return false;
        }

        auto reshape_before = std::dynamic_pointer_cast<ngraph::opset3::Reshape>(transpose->input_value(0).get_node_shared_ptr());
        if (!reshape_before || reshape_before->get_output_target_inputs(0).size() != 1) {
            return false;
        }

        auto p_shape_input = reshape_before->get_input_partial_shape(0);
        auto p_shape_reshape_before = reshape_before->get_output_partial_shape(0);
        auto p_shape_transpose = transpose->get_output_partial_shape(0);
        auto p_shape_transpose_constant = transpose_constant->get_output_partial_shape(0);
        auto p_shape_reshape_after = reshape_after->get_output_partial_shape(0);

        if (p_shape_input.is_dynamic() || p_shape_reshape_before.is_dynamic() ||
            p_shape_transpose.is_dynamic() || p_shape_reshape_after.is_dynamic() ||
            p_shape_transpose_constant.is_dynamic()) {
            return false;
        }

        auto shape_input = p_shape_input.get_shape();
        auto shape_reshape_before = p_shape_reshape_before.get_shape();
        auto shape_transpose_constant = p_shape_transpose_constant.get_shape();
        auto shape_reshape_after = p_shape_reshape_after.get_shape();

        auto transpose_constant_values = transpose_constant->get_axis_vector_val();
        if (!check_shapes(shape_input, shape_reshape_before,
                          transpose_constant_values, shape_reshape_after)) {
            return false;
        }
        
        int64_t axis = 1ul;
        int64_t group = shape_reshape_before[1];

        auto shuffle_shannels = std::make_shared<ngraph::opset3::ShuffleChannels>(reshape_before->input_value(0), axis, group);
        shuffle_shannels->set_friendly_name(reshape_after->get_friendly_name());
        ngraph::copy_runtime_info({ reshape_before, transpose, reshape_after }, shuffle_shannels);
        ngraph::replace_node(reshape_after, shuffle_shannels);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(reshape_after, matcher_name);
    register_matcher(m, callback);
}
