// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_opset3_to_opset2/convert_shuffle_channels3.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset2.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/rt_info.hpp>

using namespace ngraph;

void ngraph::pass::ConvertShuffleChannels3::convert_shuffle_channels3() {
    auto input = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto shuffle_channels = std::make_shared<::opset3::ShuffleChannels>(input);

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher &m) {
        auto shuffle_channels = std::dynamic_pointer_cast<::opset3::ShuffleChannels>(m.get_match_root());
        if (!shuffle_channels) {
            return false;
        }
        if (shuffle_channels->input_value(0).get_partial_shape().rank().is_dynamic()) {
            return false;
        }

        auto reduce_axis_const = ::opset2::Constant::create(element::i64, Shape({1}), std::vector<int64_t>{0});
        auto shuffle_axis = shuffle_channels->get_axis();
        int64_t shuffle_group = static_cast<int64_t>(shuffle_channels->get_group());
        int64_t input_rank = shuffle_channels->input_value(0).get_partial_shape().rank().get_length();
        auto original_shape = std::make_shared<::opset2::ShapeOf>(shuffle_channels->input_value(0));
        if (shuffle_axis < 0) {
            shuffle_axis += input_rank;
        }

        // calculate split sizes based on shuffle axis and avoid splits of size 0
        std::vector<int64_t> split_lengts;
        if (shuffle_axis == 0) {
            split_lengts = {1, input_rank - 1};
        } else if (shuffle_axis + 1 == input_rank) {
            split_lengts = {input_rank - 1, 1};
        } else {
            split_lengts = {shuffle_axis, 1, input_rank - shuffle_axis - 1};
        }

        // get input tensor dimensions divided into parts with help of VariadicSplit
        auto split_input_dimensions = std::make_shared<::opset2::VariadicSplit>(
                original_shape->output(0),
                ::opset2::Constant::create(element::i64, Shape({1}), std::vector<int64_t>{0}),
                ::opset2::Constant::create(element::i64, Shape({split_lengts.size()}), split_lengts));

        // calculate new dimension of the reshape. Start with two elements of {group, -1}
        ::OutputVector new_dimensions = {
                ::opset2::Constant::create(element::i64, Shape({1}), std::vector<int64_t>{shuffle_group}),
                ::opset2::Constant::create(element::i64, Shape({1}), std::vector<int64_t>{-1})};

        // add more elements to the reshape output dimensions based on shuffle_axis
        std::vector<int64_t> transpose_order;
        if (shuffle_axis == 0) {
            new_dimensions.push_back(
                    std::make_shared<::opset2::ReduceProd>(split_input_dimensions->output(1), reduce_axis_const, true));
            transpose_order = {1, 0, 2};
        } else if (shuffle_axis + 1 == input_rank) {
            new_dimensions.insert(new_dimensions.begin(),
                                  std::make_shared<::opset2::ReduceProd>(split_input_dimensions->output(0),
                                                                         reduce_axis_const, true));
            transpose_order = {0, 2, 1};
        } else {
            new_dimensions.insert(new_dimensions.begin(),
                                  std::make_shared<::opset2::ReduceProd>(split_input_dimensions->output(0),
                                                                         reduce_axis_const, true));
            new_dimensions.push_back(
                    std::make_shared<::opset2::ReduceProd>(split_input_dimensions->output(2), reduce_axis_const, true));
            transpose_order = {0, 2, 1, 3};
        }
        // reshape the tensor to a new shape
        auto new_shape = std::make_shared<::opset2::Concat>(new_dimensions, 0);
        auto reshape = std::make_shared<::opset2::Reshape>(shuffle_channels->input_value(0), new_shape, false);
        // swap dimensions appearing after splitting the "shuffle_axis" dimension into two
        auto transpose = std::make_shared<::opset2::Transpose>(reshape->output(0),
                                                               ::opset2::Constant::create(element::i64,
                                                                                          Shape({transpose_order.size()}),
                                                                                          transpose_order));
        // restore original shape
        auto reshape_back = std::make_shared<::opset2::Reshape>(transpose->output(0), original_shape->output(0), false);

        ::NodeVector new_ops = {original_shape, split_input_dimensions, transpose, reshape, reshape_back, new_shape};
        for (auto output : new_dimensions)
            new_ops.insert(new_ops.begin(), output.get_node_shared_ptr());
        reshape_back->set_friendly_name(shuffle_channels->get_friendly_name());
        ::copy_runtime_info(shuffle_channels, new_ops);
        ::replace_node(shuffle_channels, reshape_back);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(shuffle_channels, "ConvertShuffleChannels3");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}