// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/core/coordinate_diff.hpp>
#include <openvino/op/convolution.hpp>
#include <openvino/op/ops.hpp>
#include <openvino/op/parameter.hpp>
#include <utils/shape_inference/shape_inference.hpp>

#include "utils/shape_inference/static_shape.hpp"

using namespace ov;

TEST(StaticShapeInferenceTest, SelectTest) {
    // broadcast rule is AutoBroadcastType::NUMPY
    {
        auto cond = std::make_shared<op::v0::Parameter>(element::boolean, PartialShape{});
        auto ptrue = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{});
        auto pfalse = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{});
        auto select = std::make_shared<op::v1::Select>(cond, ptrue, pfalse, op::AutoBroadcastType::NUMPY);
        {
            std::vector<StaticShape> static_input_shapes = {StaticShape{}, StaticShape{4}, StaticShape{2, 4}},
                                     static_output_shapes = {StaticShape{}};
            shape_inference(select.get(), static_input_shapes, static_output_shapes);
            ASSERT_EQ(static_output_shapes[0], StaticShape({2, 4}));
        }

        {
            std::vector<StaticShape> static_input_shapes = {StaticShape{}, StaticShape{2, 4}, StaticShape{2, 4}},
                                     static_output_shapes = {StaticShape{}};
            shape_inference(select.get(), static_input_shapes, static_output_shapes);
            ASSERT_EQ(static_output_shapes[0], StaticShape({2, 4}));
        }

        {
            std::vector<StaticShape> static_input_shapes = {StaticShape{4}, StaticShape{2, 4}, StaticShape{}},
                                     static_output_shapes = {StaticShape{}};
            shape_inference(select.get(), static_input_shapes, static_output_shapes);
            ASSERT_EQ(static_output_shapes[0], StaticShape({2, 4}));
        }
    }
    // broadcast rule is op::AutoBroadcastType::PDPD
    {
        auto cond = std::make_shared<op::v0::Parameter>(element::boolean, PartialShape{});
        auto ptrue = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{});
        auto pfalse = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{});
        auto select = std::make_shared<op::v1::Select>(cond,
                                                       ptrue,
                                                       pfalse,
                                                       op::AutoBroadcastSpec{op::AutoBroadcastType::PDPD, 1});
        std::vector<StaticShape> static_input_shapes = {StaticShape{4}, StaticShape{4}, StaticShape{2, 4}},
                                 static_output_shapes = {StaticShape{}};
        shape_inference(select.get(), static_input_shapes, static_output_shapes);
        ASSERT_EQ(static_output_shapes[0], StaticShape({2, 4}));
    }
    // broadcast rule is op::AutoBroadcastType::NONE
    {
        auto cond = std::make_shared<op::v0::Parameter>(element::boolean, PartialShape{});
        auto ptrue = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{});
        auto pfalse = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{});
        auto select = std::make_shared<op::v1::Select>(cond, ptrue, pfalse, op::AutoBroadcastType::NONE);

        std::vector<StaticShape> static_input_shapes = {StaticShape{6, 4}, StaticShape{6, 4}, StaticShape{6, 4}},
                                 static_output_shapes = {StaticShape{}};
        shape_inference(select.get(), static_input_shapes, static_output_shapes);
        ASSERT_EQ(static_output_shapes[0], StaticShape({6, 4}));
    }
}
