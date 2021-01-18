// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset2.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <transformations/op_conversions/convert_shuffle_channels3.hpp>
#include <transformations/init_node_info.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

std::shared_ptr<ngraph::Function> buildInputGraph(int64_t axis, int64_t group, const ::PartialShape& p) {
    auto input = std::make_shared<::opset3::Parameter>(::element::f32, p);
    auto shuffle_channels = std::make_shared<::opset3::ShuffleChannels>(input, axis, group);
    shuffle_channels->set_friendly_name("shc");

    auto f = std::make_shared<::Function>(::NodeVector{shuffle_channels}, ::ParameterVector{input});

    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::InitNodeInfo>();
    manager.register_pass<ngraph::pass::ConvertShuffleChannels3>();
    manager.run_passes(f);
    f->validate_nodes_and_infer_types();
    return f;
}

TEST(TransformationTests, ConvertShuffleChannelsAxis0) {
    int64_t group = 4;
    auto ps = ::PartialShape{12, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()};
    std::shared_ptr<ngraph::Function> f = buildInputGraph(0, group, ps), f_ref(nullptr);
    ASSERT_NO_THROW(check_rt_info(f));

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
    auto transpose = std::make_shared<::opset2::Transpose>(reshape->output(0),
                                                           ::opset2::Constant::create(element::i64, Shape({3}),
                                                                                      {1, 0, 2}));
    auto reshape_back = std::make_shared<::opset2::Reshape>(transpose->output(0), original_shape->output(0), false);

    f_ref = std::make_shared<::Function>(::NodeVector{reshape_back}, ::ParameterVector{input});

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;

    auto result_node_of_converted_f = f->get_output_op(0);
    auto output_node = result_node_of_converted_f->input(0).get_source_output().get_node_shared_ptr();
    ASSERT_TRUE(output_node->get_friendly_name() == "shc") << "ConvertShuffleChannels3 should keep output names.\n";
}

TEST(TransformationTests, ConvertShuffleChannelsAxis1) {
    int64_t group = 4;
    auto ps = ::PartialShape{Dimension::dynamic(), 12, Dimension::dynamic(), Dimension::dynamic()};
    std::shared_ptr<ngraph::Function> f = buildInputGraph(1, group, ps), f_ref(nullptr);
    ASSERT_NO_THROW(check_rt_info(f));

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
    auto transpose = std::make_shared<::opset2::Transpose>(reshape->output(0),
                                                           ::opset2::Constant::create(element::i64, Shape({4}),
                                                                                      {0, 2, 1, 3}));
    auto reshape_back = std::make_shared<::opset2::Reshape>(transpose->output(0), original_shape->output(0), false);

    f_ref = std::make_shared<::Function>(::NodeVector{reshape_back}, ::ParameterVector{input});

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;

    auto result_node_of_converted_f = f->get_output_op(0);
    auto output_node = result_node_of_converted_f->input(0).get_source_output().get_node_shared_ptr();
    ASSERT_TRUE(output_node->get_friendly_name() == "shc") << "ConvertShuffleChannels3 should keep output names.\n";
}

TEST(TransformationTests, ConvertShuffleChannelsAxis2) {
    int64_t group = 4;
    auto ps = ::PartialShape{Dimension::dynamic(), Dimension::dynamic(), 12, Dimension::dynamic()};
    std::shared_ptr<ngraph::Function> f = buildInputGraph(2, group, ps), f_ref(nullptr);
    ASSERT_NO_THROW(check_rt_info(f));

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
    auto transpose = std::make_shared<::opset2::Transpose>(reshape->output(0),
                                                           ::opset2::Constant::create(element::i64, Shape({4}),
                                                                                      {0, 2, 1, 3}));
    auto reshape_back = std::make_shared<::opset2::Reshape>(transpose->output(0), original_shape->output(0), false);

    f_ref = std::make_shared<::Function>(::NodeVector{reshape_back}, ::ParameterVector{input});

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;

    auto result_node_of_converted_f = f->get_output_op(0);
    auto output_node = result_node_of_converted_f->input(0).get_source_output().get_node_shared_ptr();
    ASSERT_TRUE(output_node->get_friendly_name() == "shc") << "ConvertShuffleChannels3 should keep output names.\n";
}

TEST(TransformationTests, ConvertShuffleChannelsLastAxis) {
    int64_t group = 4;
    auto ps = ::PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), 12};
    std::shared_ptr<ngraph::Function> f = buildInputGraph(-1, group, ps), f_ref(nullptr);
    ASSERT_NO_THROW(check_rt_info(f));

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
    auto transpose = std::make_shared<::opset2::Transpose>(reshape->output(0),
                                                           ::opset2::Constant::create(element::i64, Shape({3}),
                                                                                      {0, 2, 1}));
    auto reshape_back = std::make_shared<::opset2::Reshape>(transpose->output(0), original_shape->output(0), false);

    f_ref = std::make_shared<::Function>(::NodeVector{reshape_back}, ::ParameterVector{input});

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;

    auto result_node_of_converted_f = f->get_output_op(0);
    auto output_node = result_node_of_converted_f->input(0).get_source_output().get_node_shared_ptr();
    ASSERT_TRUE(output_node->get_friendly_name() == "shc") << "ConvertShuffleChannels3 should keep output names.\n";
}