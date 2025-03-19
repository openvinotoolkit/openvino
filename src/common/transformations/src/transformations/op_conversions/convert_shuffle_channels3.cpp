// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_shuffle_channels3.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/shuffle_channels.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using namespace ov;

ov::pass::ConvertShuffleChannels3::ConvertShuffleChannels3() {
    MATCHER_SCOPE(ConvertShuffleChannels3);
    auto shuffle_channels = pattern::wrap_type<ov::op::v0::ShuffleChannels>();

    matcher_pass_callback callback = [this](pattern::Matcher& m) {
        auto shuffle_channels = ov::as_type_ptr<ov::op::v0::ShuffleChannels>(m.get_match_root());
        if (!shuffle_channels || transformation_callback(shuffle_channels)) {
            return false;
        }
        if (shuffle_channels->input_value(0).get_partial_shape().rank().is_dynamic()) {
            return false;
        }

        auto reduce_axis_const = ov::op::v0::Constant::create(element::i64, Shape({1}), std::vector<int64_t>{0});
        auto shuffle_axis = shuffle_channels->get_axis();
        int64_t shuffle_group = static_cast<int64_t>(shuffle_channels->get_group());
        int64_t input_rank = shuffle_channels->input_value(0).get_partial_shape().rank().get_length();
        auto original_shape = std::make_shared<ov::op::v0::ShapeOf>(shuffle_channels->input_value(0));
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
        auto split_input_dimensions = std::make_shared<ov::op::v1::VariadicSplit>(
            original_shape->output(0),
            ov::op::v0::Constant::create(element::i64, Shape({1}), std::vector<int64_t>{0}),
            ov::op::v0::Constant::create(element::i64, Shape({split_lengts.size()}), split_lengts));

        // calculate new dimension of the reshape. Start with two elements of {group, -1}
        ::OutputVector new_dimensions = {
            ov::op::v0::Constant::create(element::i64, Shape({1}), std::vector<int64_t>{shuffle_group}),
            ov::op::v0::Constant::create(element::i64, Shape({1}), std::vector<int64_t>{-1})};

        // add more elements to the reshape output dimensions based on shuffle_axis
        std::vector<int64_t> transpose_order;
        if (shuffle_axis == 0) {
            new_dimensions.push_back(
                std::make_shared<ov::op::v1::ReduceProd>(split_input_dimensions->output(1), reduce_axis_const, true));
            transpose_order = {1, 0, 2};
        } else if (shuffle_axis + 1 == input_rank) {
            new_dimensions.insert(
                new_dimensions.begin(),
                std::make_shared<ov::op::v1::ReduceProd>(split_input_dimensions->output(0), reduce_axis_const, true));
            transpose_order = {0, 2, 1};
        } else {
            new_dimensions.insert(
                new_dimensions.begin(),
                std::make_shared<ov::op::v1::ReduceProd>(split_input_dimensions->output(0), reduce_axis_const, true));
            new_dimensions.push_back(
                std::make_shared<ov::op::v1::ReduceProd>(split_input_dimensions->output(2), reduce_axis_const, true));
            transpose_order = {0, 2, 1, 3};
        }
        // reshape the tensor to a new shape
        auto new_shape = std::make_shared<ov::op::v0::Concat>(new_dimensions, 0);
        auto reshape = std::make_shared<ov::op::v1::Reshape>(shuffle_channels->input_value(0), new_shape, false);
        // swap dimensions appearing after splitting the "shuffle_axis" dimension into two
        auto transpose = std::make_shared<ov::op::v1::Transpose>(
            reshape->output(0),
            ov::op::v0::Constant::create(element::i64, Shape({transpose_order.size()}), transpose_order));
        // restore original shape
        auto reshape_back =
            std::make_shared<ov::op::v1::Reshape>(transpose->output(0), original_shape->output(0), false);

        ::NodeVector new_ops = {original_shape, split_input_dimensions, transpose, reshape, reshape_back, new_shape};
        for (auto output : new_dimensions)
            new_ops.insert(new_ops.begin(), output.get_node_shared_ptr());
        reshape_back->set_friendly_name(shuffle_channels->get_friendly_name());
        ::copy_runtime_info(shuffle_channels, new_ops);
        ::replace_node(shuffle_channels, reshape_back);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(shuffle_channels, matcher_name);
    register_matcher(m, callback);
}
