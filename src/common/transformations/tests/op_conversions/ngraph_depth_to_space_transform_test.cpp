// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/depth_to_space.hpp"
#include "openvino/op/space_to_depth.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/op_conversions/convert_depth_to_space.hpp"
#include "transformations/op_conversions/convert_space_to_depth.hpp"
using namespace ov;
using namespace testing;

TEST(TransformationTests, TestDepthToSpaceTransformBlockFirst) {
    auto input = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 12, 1080, 1616});
    std::shared_ptr<ov::Model> f(nullptr);

    {
        auto depth_to_space =
            std::make_shared<op::v0::DepthToSpace>(input, op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, 2);
        f = std::make_shared<ov::Model>(ov::NodeVector{depth_to_space}, ParameterVector{input});
        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::ConvertDepthToSpace>();
        m.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    auto consumers = input->output(0).get_target_inputs();
    ASSERT_EQ(consumers.size(), 1);

    auto reshape_begin = consumers.begin()->get_node();
    auto shape_begin =
        ov::as_type_ptr<ov::op::v0::Constant>(reshape_begin->input(1).get_source_output().get_node_shared_ptr());
    std::vector<int64_t> shape_begin_value = shape_begin->get_vector<int64_t>();
    std::vector<int64_t> shape_begin_value_ref{1, 2, 2, 3, 1080, 1616};
    ASSERT_EQ(shape_begin_value, shape_begin_value_ref);

    consumers = reshape_begin->output(0).get_target_inputs();
    ASSERT_EQ(consumers.size(), 1);

    auto transpose = consumers.begin()->get_node();
    auto order = ov::as_type_ptr<ov::op::v0::Constant>(transpose->input(1).get_source_output().get_node_shared_ptr());
    std::vector<int64_t> order_value = order->get_vector<int64_t>();
    std::vector<int64_t> order_value_ref{0, 3, 4, 1, 5, 2};
    ASSERT_EQ(order_value, order_value_ref);

    consumers = transpose->output(0).get_target_inputs();
    auto reshape_end = consumers.begin()->get_node();
    auto shape_end =
        ov::as_type_ptr<ov::op::v0::Constant>(reshape_end->input(1).get_source_output().get_node_shared_ptr());
    std::vector<int64_t> shape_end_value = shape_end->get_vector<int64_t>();
    std::vector<int64_t> shape_end_value_ref{1, 3, 2 * 1080, 2 * 1616};
    ASSERT_EQ(shape_end_value, shape_end_value_ref);
}

TEST(TransformationTests, TestDepthToSpaceTransformDepthFirst) {
    auto input = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 12, 1080, 1616});
    std::shared_ptr<ov::Model> f(nullptr);

    {
        auto depth_to_space =
            std::make_shared<op::v0::DepthToSpace>(input, op::v0::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST, 2);
        f = std::make_shared<ov::Model>(ov::NodeVector{depth_to_space}, ParameterVector{input});
        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::ConvertDepthToSpace>();
        m.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    auto consumers = input->output(0).get_target_inputs();
    ASSERT_EQ(consumers.size(), 1);

    auto reshape_begin = consumers.begin()->get_node();
    auto shape_begin =
        ov::as_type_ptr<ov::op::v0::Constant>(reshape_begin->input(1).get_source_output().get_node_shared_ptr());
    std::vector<int64_t> shape_begin_value = shape_begin->get_vector<int64_t>();
    std::vector<int64_t> shape_begin_value_ref{1, 3, 2, 2, 1080, 1616};
    ASSERT_EQ(shape_begin_value, shape_begin_value_ref);

    consumers = reshape_begin->output(0).get_target_inputs();
    ASSERT_EQ(consumers.size(), 1);

    auto transpose = consumers.begin()->get_node();
    auto order = ov::as_type_ptr<ov::op::v0::Constant>(transpose->input(1).get_source_output().get_node_shared_ptr());
    std::vector<int64_t> order_value = order->get_vector<int64_t>();
    std::vector<int64_t> order_value_ref{0, 1, 4, 2, 5, 3};
    ASSERT_EQ(order_value, order_value_ref);

    consumers = transpose->output(0).get_target_inputs();
    auto reshape_end = consumers.begin()->get_node();
    auto shape_end =
        ov::as_type_ptr<ov::op::v0::Constant>(reshape_end->input(1).get_source_output().get_node_shared_ptr());
    std::vector<int64_t> shape_end_value = shape_end->get_vector<int64_t>();
    std::vector<int64_t> shape_end_value_ref{1, 3, 2 * 1080, 2 * 1616};
    ASSERT_EQ(shape_end_value, shape_end_value_ref);
}

TEST(TransformationTests, TestSpaceToDepthTransformBlockFirst) {
    auto input = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 12, 1080, 1616});
    std::shared_ptr<ov::Model> f(nullptr);

    {
        auto space_to_depth =
            std::make_shared<op::v0::SpaceToDepth>(input, op::v0::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST, 2);
        f = std::make_shared<ov::Model>(ov::NodeVector{space_to_depth}, ParameterVector{input});
        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::ConvertSpaceToDepth>();
        m.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    auto consumers = input->output(0).get_target_inputs();
    ASSERT_EQ(consumers.size(), 1);

    auto reshape_begin = consumers.begin()->get_node();
    auto shape_begin =
        ov::as_type_ptr<ov::op::v0::Constant>(reshape_begin->input(1).get_source_output().get_node_shared_ptr());
    std::vector<int64_t> shape_begin_value = shape_begin->get_vector<int64_t>();
    std::vector<int64_t> shape_begin_value_ref{1, 12, 1080 / 2, 2, 1616 / 2, 2};
    ASSERT_EQ(shape_begin_value, shape_begin_value_ref);

    consumers = reshape_begin->output(0).get_target_inputs();
    ASSERT_EQ(consumers.size(), 1);

    auto transpose = consumers.begin()->get_node();
    auto order = ov::as_type_ptr<ov::op::v0::Constant>(transpose->input(1).get_source_output().get_node_shared_ptr());
    std::vector<int64_t> order_value = order->get_vector<int64_t>();
    std::vector<int64_t> order_value_ref{0, 3, 5, 1, 2, 4};
    ASSERT_EQ(order_value, order_value_ref);

    consumers = transpose->output(0).get_target_inputs();
    auto reshape_end = consumers.begin()->get_node();
    auto shape_end =
        ov::as_type_ptr<ov::op::v0::Constant>(reshape_end->input(1).get_source_output().get_node_shared_ptr());
    std::vector<int64_t> shape_end_value = shape_end->get_vector<int64_t>();
    std::vector<int64_t> shape_end_value_ref{1, 12 * 4, 1080 / 2, 1616 / 2};
    ASSERT_EQ(shape_end_value, shape_end_value_ref);
}

TEST(TransformationTests, TestSpaceToDepthTransformDepthFirst) {
    auto input = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 12, 1080, 1616});
    std::shared_ptr<ov::Model> f(nullptr);

    {
        auto space_to_depth =
            std::make_shared<op::v0::SpaceToDepth>(input, op::v0::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST, 2);
        f = std::make_shared<ov::Model>(ov::NodeVector{space_to_depth}, ParameterVector{input});
        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::ConvertSpaceToDepth>();
        m.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    auto consumers = input->output(0).get_target_inputs();
    ASSERT_EQ(consumers.size(), 1);

    auto reshape_begin = consumers.begin()->get_node();
    auto shape_begin =
        ov::as_type_ptr<ov::op::v0::Constant>(reshape_begin->input(1).get_source_output().get_node_shared_ptr());
    std::vector<int64_t> shape_begin_value = shape_begin->get_vector<int64_t>();
    std::vector<int64_t> shape_begin_value_ref{1, 12, 1080 / 2, 2, 1616 / 2, 2};
    ASSERT_EQ(shape_begin_value, shape_begin_value_ref);

    consumers = reshape_begin->output(0).get_target_inputs();
    ASSERT_EQ(consumers.size(), 1);

    auto transpose = consumers.begin()->get_node();
    auto order = ov::as_type_ptr<ov::op::v0::Constant>(transpose->input(1).get_source_output().get_node_shared_ptr());
    std::vector<int64_t> order_value = order->get_vector<int64_t>();
    std::vector<int64_t> order_value_ref{0, 1, 3, 5, 2, 4};
    ASSERT_EQ(order_value, order_value_ref);

    consumers = transpose->output(0).get_target_inputs();
    auto reshape_end = consumers.begin()->get_node();
    auto shape_end =
        ov::as_type_ptr<ov::op::v0::Constant>(reshape_end->input(1).get_source_output().get_node_shared_ptr());
    std::vector<int64_t> shape_end_value = shape_end->get_vector<int64_t>();
    std::vector<int64_t> shape_end_value_ref{1, 12 * 4, 1080 / 2, 1616 / 2};
    ASSERT_EQ(shape_end_value, shape_end_value_ref);
}

TEST(TransformationTests, TestSpaceToDepthDynamic) {
    auto input = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    std::shared_ptr<ov::Model> f(nullptr);

    {
        auto space_to_depth =
            std::make_shared<op::v0::SpaceToDepth>(input, op::v0::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST, 2);
        f = std::make_shared<ov::Model>(ov::NodeVector{space_to_depth}, ParameterVector{input});
        pass::Manager m;
        m.register_pass<ov::pass::ConvertSpaceToDepth>();
        OV_ASSERT_NO_THROW(m.run_passes(f));
    }
}

TEST(TransformationTests, TestDepthToSpaceDynamic) {
    auto input = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    std::shared_ptr<ov::Model> f(nullptr);

    {
        auto depth_to_space =
            std::make_shared<op::v0::DepthToSpace>(input, op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, 2);
        f = std::make_shared<ov::Model>(ov::NodeVector{depth_to_space}, ParameterVector{input});
        pass::Manager m;
        m.register_pass<ov::pass::ConvertDepthToSpace>();
        OV_ASSERT_NO_THROW(m.run_passes(f));
    }
}
