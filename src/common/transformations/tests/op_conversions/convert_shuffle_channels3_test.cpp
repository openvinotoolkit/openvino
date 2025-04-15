// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_shuffle_channels3.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset2.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"

using namespace testing;
using namespace ov;

std::shared_ptr<ov::Model> buildInputGraph(int64_t axis, int64_t group, const ::PartialShape& p) {
    auto input = std::make_shared<::opset3::Parameter>(::element::f32, p);
    auto shuffle_channels = std::make_shared<::opset3::ShuffleChannels>(input, axis, group);
    return std::make_shared<::Model>(::NodeVector{shuffle_channels}, ::ParameterVector{input});
}

TEST_F(TransformationTestsF, ConvertShuffleChannelsAxis0) {
    int64_t group = 4;
    auto ps = ::PartialShape{12, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()};
    model = buildInputGraph(0, group, ps);
    manager.register_pass<ov::pass::ConvertShuffleChannels3>();

    auto input = std::make_shared<::opset3::Parameter>(::element::f32, ps);

    auto reduce_axis_const = ::opset2::Constant::create(element::i64, Shape({1}), std::vector<int64_t>{0});
    auto original_shape = std::make_shared<::opset2::ShapeOf>(input->output(0));
    auto split_input_dimensions = std::make_shared<::opset2::VariadicSplit>(
        original_shape->output(0),
        ::opset2::Constant::create(element::i64, Shape({1}), std::vector<int64_t>{0}),
        ::opset2::Constant::create(element::i64, Shape({2}), {1, 3}));

    ::OutputVector new_dims = {
        ::opset2::Constant::create(element::i64, Shape({1}), std::vector<int64_t>{group}),
        ::opset2::Constant::create(element::i64, Shape({1}), std::vector<int64_t>{-1}),
        std::make_shared<::opset2::ReduceProd>(split_input_dimensions->output(1), reduce_axis_const, true)};

    auto new_shape = std::make_shared<::opset2::Concat>(new_dims, 0);
    auto reshape = std::make_shared<::opset2::Reshape>(input->output(0), new_shape, false);
    auto transpose =
        std::make_shared<::opset2::Transpose>(reshape->output(0),
                                              ::opset2::Constant::create(element::i64, Shape({3}), {1, 0, 2}));
    auto reshape_back = std::make_shared<::opset2::Reshape>(transpose->output(0), original_shape->output(0), false);

    model_ref = std::make_shared<::Model>(::NodeVector{reshape_back}, ::ParameterVector{input});
}

TEST_F(TransformationTestsF, ConvertShuffleChannelsAxis1) {
    int64_t group = 4;
    auto ps = ::PartialShape{Dimension::dynamic(), 12, Dimension::dynamic(), Dimension::dynamic()};
    model = buildInputGraph(1, group, ps);
    manager.register_pass<ov::pass::ConvertShuffleChannels3>();

    auto input = std::make_shared<::opset3::Parameter>(::element::f32, ps);

    auto reduce_axis_const = ::opset2::Constant::create(element::i64, Shape({1}), std::vector<int64_t>{0});
    auto original_shape = std::make_shared<::opset2::ShapeOf>(input->output(0));
    auto split_input_dimensions = std::make_shared<::opset2::VariadicSplit>(
        original_shape->output(0),
        ::opset2::Constant::create(element::i64, Shape({1}), std::vector<int64_t>{0}),
        ::opset2::Constant::create(element::i64, Shape({3}), {1, 1, 2}));

    ::OutputVector new_dims = {
        std::make_shared<::opset2::ReduceProd>(split_input_dimensions->output(0), reduce_axis_const, true),
        ::opset2::Constant::create(element::i64, Shape({1}), std::vector<int64_t>{group}),
        ::opset2::Constant::create(element::i64, Shape({1}), std::vector<int64_t>{-1}),
        std::make_shared<::opset2::ReduceProd>(split_input_dimensions->output(2), reduce_axis_const, true)};

    auto new_shape = std::make_shared<::opset2::Concat>(new_dims, 0);
    auto reshape = std::make_shared<::opset2::Reshape>(input->output(0), new_shape, false);
    auto transpose =
        std::make_shared<::opset2::Transpose>(reshape->output(0),
                                              ::opset2::Constant::create(element::i64, Shape({4}), {0, 2, 1, 3}));
    auto reshape_back = std::make_shared<::opset2::Reshape>(transpose->output(0), original_shape->output(0), false);

    model_ref = std::make_shared<::Model>(::NodeVector{reshape_back}, ::ParameterVector{input});
}

TEST_F(TransformationTestsF, ConvertShuffleChannelsAxis2) {
    int64_t group = 4;
    auto ps = ::PartialShape{Dimension::dynamic(), Dimension::dynamic(), 12, Dimension::dynamic()};
    model = buildInputGraph(2, group, ps);
    manager.register_pass<ov::pass::ConvertShuffleChannels3>();

    auto input = std::make_shared<::opset3::Parameter>(::element::f32, ps);

    auto reduce_axis_const = ::opset2::Constant::create(element::i64, Shape({1}), std::vector<int64_t>{0});
    auto original_shape = std::make_shared<::opset2::ShapeOf>(input->output(0));
    auto split_input_dimensions = std::make_shared<::opset2::VariadicSplit>(
        original_shape->output(0),
        ::opset2::Constant::create(element::i64, Shape({1}), std::vector<int64_t>{0}),
        ::opset2::Constant::create(element::i64, Shape({3}), {2, 1, 1}));

    ::OutputVector new_dims = {
        std::make_shared<::opset2::ReduceProd>(split_input_dimensions->output(0), reduce_axis_const, true),
        ::opset2::Constant::create(element::i64, Shape({1}), std::vector<int64_t>{group}),
        ::opset2::Constant::create(element::i64, Shape({1}), std::vector<int64_t>{-1}),
        std::make_shared<::opset2::ReduceProd>(split_input_dimensions->output(2), reduce_axis_const, true)};

    auto new_shape = std::make_shared<::opset2::Concat>(new_dims, 0);
    auto reshape = std::make_shared<::opset2::Reshape>(input->output(0), new_shape, false);
    auto transpose =
        std::make_shared<::opset2::Transpose>(reshape->output(0),
                                              ::opset2::Constant::create(element::i64, Shape({4}), {0, 2, 1, 3}));
    auto reshape_back = std::make_shared<::opset2::Reshape>(transpose->output(0), original_shape->output(0), false);

    model_ref = std::make_shared<::Model>(::NodeVector{reshape_back}, ::ParameterVector{input});
}

TEST_F(TransformationTestsF, ConvertShuffleChannelsLastAxis) {
    int64_t group = 4;
    auto ps = ::PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), 12};
    model = buildInputGraph(-1, group, ps);
    manager.register_pass<ov::pass::ConvertShuffleChannels3>();

    auto input = std::make_shared<::opset3::Parameter>(::element::f32, ps);

    auto reduce_axis_const = ::opset2::Constant::create(element::i64, Shape({1}), std::vector<int64_t>{0});
    auto original_shape = std::make_shared<::opset2::ShapeOf>(input->output(0));
    auto split_input_dimensions = std::make_shared<::opset2::VariadicSplit>(
        original_shape->output(0),
        ::opset2::Constant::create(element::i64, Shape({1}), std::vector<int64_t>{0}),
        ::opset2::Constant::create(element::i64, Shape({2}), {3, 1}));

    ::OutputVector new_dims = {
        std::make_shared<::opset2::ReduceProd>(split_input_dimensions->output(0), reduce_axis_const, true),
        ::opset2::Constant::create(element::i64, Shape({1}), std::vector<int64_t>{group}),
        ::opset2::Constant::create(element::i64, Shape({1}), std::vector<int64_t>{-1})};

    auto new_shape = std::make_shared<::opset2::Concat>(new_dims, 0);
    auto reshape = std::make_shared<::opset2::Reshape>(input->output(0), new_shape, false);
    auto transpose =
        std::make_shared<::opset2::Transpose>(reshape->output(0),
                                              ::opset2::Constant::create(element::i64, Shape({3}), {0, 2, 1}));
    auto reshape_back = std::make_shared<::opset2::Reshape>(transpose->output(0), original_shape->output(0), false);

    model_ref = std::make_shared<::Model>(::NodeVector{reshape_back}, ::ParameterVector{input});
}
