// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/shape_inference/static_shape.hpp"
#include <gtest/gtest.h>
#include <openvino/op/ops.hpp>
#include <openvino/op/parameter.hpp>
#include <shuffle_channels_shape_inference.hpp>

using namespace ov;

TEST(StaticShapeInferenceTest, ShuffleChannelsTest) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1});
    const auto axis = -1;
    const auto group = 3;
    const auto shuffle_channels = std::make_shared<ov::op::v0::ShuffleChannels>(data, axis, group);

    std::vector<PartialShape> input_shapes = {PartialShape{5, 4, 9}};
    std::vector<PartialShape> output_shapes = {PartialShape{}};
    shape_infer(shuffle_channels.get(), input_shapes, output_shapes);

    ASSERT_EQ(output_shapes[0], input_shapes[0]);

    std::vector<StaticShape> static_input_shapes = {StaticShape{5, 4, 9}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{}};
    shape_infer(shuffle_channels.get(), static_input_shapes, static_output_shapes);

    ASSERT_EQ(static_output_shapes[0], static_input_shapes[0]);
}