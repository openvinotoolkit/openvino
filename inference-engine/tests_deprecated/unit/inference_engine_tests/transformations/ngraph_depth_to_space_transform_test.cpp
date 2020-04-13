// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "tests_common.hpp"
#include <string>
#include <sstream>
#include <fstream>
#include <memory>
#include <map>

#include <ngraph/function.hpp>
#include <ngraph/op/fused/depth_to_space.hpp>
#include <ngraph/op/fused/space_to_depth.hpp>
#include <transformations/convert_depth_to_space.hpp>
#include <transformations/convert_space_to_depth.hpp>

using namespace testing;

class DepthAndSpaceTransformTests : public TestsCommon {};

TEST_F(DepthAndSpaceTransformTests, TestDepthToSpaceTransformBlockFirst) {
    auto input = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, ngraph::Shape{1, 12, 1080, 1616});
    std::shared_ptr<ngraph::Function> f(nullptr);

    {
        auto depth_to_space = std::make_shared<ngraph::op::DepthToSpace>(input, ngraph::op::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, 2);
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{depth_to_space}, ngraph::ParameterVector{input});
        ngraph::pass::ConvertDepthToSpace().run_on_function(f);
    }

    auto consumers = input->output(0).get_target_inputs();
    ASSERT_TRUE(consumers.size() == 1);

    auto reshape_begin = consumers.begin()->get_node();
    auto shape_begin = std::dynamic_pointer_cast<ngraph::op::Constant>(reshape_begin->input(1).get_source_output().get_node_shared_ptr());
    std::vector<int64_t> shape_begin_value = shape_begin->get_vector<int64_t>();
    std::vector<int64_t> shape_begin_value_ref{1, 2, 2, 3, 1080, 1616};
    compare(shape_begin_value, shape_begin_value_ref);

    consumers = reshape_begin->output(0).get_target_inputs();
    ASSERT_TRUE(consumers.size() == 1);

    auto transpose = consumers.begin()->get_node();
    auto order = std::dynamic_pointer_cast<ngraph::op::Constant>(transpose->input(1).get_source_output().get_node_shared_ptr());
    std::vector<int64_t> order_value = order->get_vector<int64_t>();
    std::vector<int64_t> order_value_ref{0, 3, 4, 1, 5, 2};
    compare(order_value, order_value_ref);

    consumers = transpose->output(0).get_target_inputs();
    auto reshape_end = consumers.begin()->get_node();
    auto shape_end = std::dynamic_pointer_cast<ngraph::op::Constant>(reshape_end->input(1).get_source_output().get_node_shared_ptr());
    std::vector<int64_t> shape_end_value = shape_end->get_vector<int64_t>();
    std::vector<int64_t> shape_end_value_ref{1, 3, 2 * 1080, 2 * 1616};
    compare(shape_end_value, shape_end_value_ref);
}

TEST_F(DepthAndSpaceTransformTests, TestDepthToSpaceTransformDepthFirst) {
    auto input = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, ngraph::Shape{1, 12, 1080, 1616});
    std::shared_ptr<ngraph::Function> f(nullptr);

    {
        auto depth_to_space = std::make_shared<ngraph::op::DepthToSpace>(input, ngraph::op::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST, 2);
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{depth_to_space}, ngraph::ParameterVector{input});
        ngraph::pass::ConvertDepthToSpace().run_on_function(f);
    }

    auto consumers = input->output(0).get_target_inputs();
    ASSERT_TRUE(consumers.size() == 1);

    auto reshape_begin = consumers.begin()->get_node();
    auto shape_begin = std::dynamic_pointer_cast<ngraph::op::Constant>(reshape_begin->input(1).get_source_output().get_node_shared_ptr());
    std::vector<int64_t> shape_begin_value = shape_begin->get_vector<int64_t>();
    std::vector<int64_t> shape_begin_value_ref{1, 3, 2, 2, 1080, 1616};
    compare(shape_begin_value, shape_begin_value_ref);

    consumers = reshape_begin->output(0).get_target_inputs();
    ASSERT_TRUE(consumers.size() == 1);

    auto transpose = consumers.begin()->get_node();
    auto order = std::dynamic_pointer_cast<ngraph::op::Constant>(transpose->input(1).get_source_output().get_node_shared_ptr());
    std::vector<int64_t> order_value = order->get_vector<int64_t>();
    std::vector<int64_t> order_value_ref{0, 1, 4, 2, 5, 3};
    compare(order_value, order_value_ref);

    consumers = transpose->output(0).get_target_inputs();
    auto reshape_end = consumers.begin()->get_node();
    auto shape_end = std::dynamic_pointer_cast<ngraph::op::Constant>(reshape_end->input(1).get_source_output().get_node_shared_ptr());
    std::vector<int64_t> shape_end_value = shape_end->get_vector<int64_t>();
    std::vector<int64_t> shape_end_value_ref{1, 3, 2 * 1080, 2 * 1616};
    compare(shape_end_value, shape_end_value_ref);
}

TEST_F(DepthAndSpaceTransformTests, TestSpaceToDepthTransformBlockFirst) {
    auto input = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, ngraph::Shape{1, 12, 1080, 1616});
    std::shared_ptr<ngraph::Function> f(nullptr);

    {
        auto space_to_depth = std::make_shared<ngraph::op::SpaceToDepth>(input, ngraph::op::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST, 2);
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{space_to_depth}, ngraph::ParameterVector{input});
        ngraph::pass::ConvertSpaceToDepth().run_on_function(f);
    }

    auto consumers = input->output(0).get_target_inputs();
    ASSERT_TRUE(consumers.size() == 1);

    auto reshape_begin = consumers.begin()->get_node();
    auto shape_begin = std::dynamic_pointer_cast<ngraph::op::Constant>(reshape_begin->input(1).get_source_output().get_node_shared_ptr());
    std::vector<int64_t> shape_begin_value = shape_begin->get_vector<int64_t>();
    std::vector<int64_t> shape_begin_value_ref{1, 12, 1080 / 2, 2, 1616 / 2, 2};
    compare(shape_begin_value, shape_begin_value_ref);

    consumers = reshape_begin->output(0).get_target_inputs();
    ASSERT_TRUE(consumers.size() == 1);

    auto transpose = consumers.begin()->get_node();
    auto order = std::dynamic_pointer_cast<ngraph::op::Constant>(transpose->input(1).get_source_output().get_node_shared_ptr());
    std::vector<int64_t> order_value = order->get_vector<int64_t>();
    std::vector<int64_t> order_value_ref{0, 3, 5, 1, 2, 4};
    compare(order_value, order_value_ref);

    consumers = transpose->output(0).get_target_inputs();
    auto reshape_end = consumers.begin()->get_node();
    auto shape_end = std::dynamic_pointer_cast<ngraph::op::Constant>(reshape_end->input(1).get_source_output().get_node_shared_ptr());
    std::vector<int64_t> shape_end_value = shape_end->get_vector<int64_t>();
    std::vector<int64_t> shape_end_value_ref{1, 12 * 4, 1080 / 2, 1616 / 2};
    compare(shape_end_value, shape_end_value_ref);
}

TEST_F(DepthAndSpaceTransformTests, TestSpaceToDepthTransformDepthFirst) {
    auto input = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, ngraph::Shape{1, 12, 1080, 1616});
    std::shared_ptr<ngraph::Function> f(nullptr);

    {
        auto space_to_depth = std::make_shared<ngraph::op::SpaceToDepth>(input, ngraph::op::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST, 2);
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{space_to_depth}, ngraph::ParameterVector{input});
        ngraph::pass::ConvertSpaceToDepth().run_on_function(f);
    }

    auto consumers = input->output(0).get_target_inputs();
    ASSERT_TRUE(consumers.size() == 1);

    auto reshape_begin = consumers.begin()->get_node();
    auto shape_begin = std::dynamic_pointer_cast<ngraph::op::Constant>(reshape_begin->input(1).get_source_output().get_node_shared_ptr());
    std::vector<int64_t> shape_begin_value = shape_begin->get_vector<int64_t>();
    std::vector<int64_t> shape_begin_value_ref{1, 12, 1080 / 2, 2, 1616 / 2, 2};
    compare(shape_begin_value, shape_begin_value_ref);

    consumers = reshape_begin->output(0).get_target_inputs();
    ASSERT_TRUE(consumers.size() == 1);

    auto transpose = consumers.begin()->get_node();
    auto order = std::dynamic_pointer_cast<ngraph::op::Constant>(transpose->input(1).get_source_output().get_node_shared_ptr());
    std::vector<int64_t> order_value = order->get_vector<int64_t>();
    std::vector<int64_t> order_value_ref{0, 1, 3, 5, 2, 4};
    compare(order_value, order_value_ref);

    consumers = transpose->output(0).get_target_inputs();
    auto reshape_end = consumers.begin()->get_node();
    auto shape_end = std::dynamic_pointer_cast<ngraph::op::Constant>(reshape_end->input(1).get_source_output().get_node_shared_ptr());
    std::vector<int64_t> shape_end_value = shape_end->get_vector<int64_t>();
    std::vector<int64_t> shape_end_value_ref{1, 12 * 4, 1080 / 2, 1616 / 2};
    compare(shape_end_value, shape_end_value_ref);
}