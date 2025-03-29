// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;

TEST(StaticShapeInferenceTest, SelectTestBCastModeNUMPY) {
    auto cond = std::make_shared<op::v0::Parameter>(element::boolean, PartialShape::dynamic());
    auto ptrue = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto pfalse = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto select = std::make_shared<op::v1::Select>(cond, ptrue, pfalse, op::AutoBroadcastType::NUMPY);
    {
        std::vector<StaticShape> static_input_shapes = {StaticShape{}, StaticShape{4}, StaticShape{2, 4}};
        const auto static_output_shapes = shape_inference(select.get(), static_input_shapes);
        EXPECT_EQ(static_output_shapes[0], StaticShape({2, 4}));
    }

    {
        std::vector<StaticShape> static_input_shapes = {StaticShape{}, StaticShape{2, 4}, StaticShape{2, 4}};
        const auto static_output_shapes = shape_inference(select.get(), static_input_shapes);
        EXPECT_EQ(static_output_shapes[0], StaticShape({2, 4}));
    }

    {
        std::vector<StaticShape> static_input_shapes = {StaticShape{4}, StaticShape{2, 4}, StaticShape{4}};
        const auto static_output_shapes = shape_inference(select.get(), static_input_shapes);
        EXPECT_EQ(static_output_shapes[0], StaticShape({2, 4}));
    }
}

TEST(StaticShapeInferenceTest, SelectTestBCastModePDPD) {
    auto cond = std::make_shared<op::v0::Parameter>(element::boolean, PartialShape::dynamic());
    auto ptrue = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto pfalse = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto select =
        std::make_shared<op::v1::Select>(cond, ptrue, pfalse, op::AutoBroadcastSpec{op::AutoBroadcastType::PDPD, 1});
    std::vector<StaticShape> static_input_shapes = {StaticShape{4}, StaticShape{2, 4}, StaticShape{4}};
    const auto static_output_shapes = shape_inference(select.get(), static_input_shapes);
    ASSERT_EQ(static_output_shapes[0], StaticShape({2, 4}));
}

TEST(StaticShapeInferenceTest, SelectTestBCastModeNone) {
    auto cond = std::make_shared<op::v0::Parameter>(element::boolean, PartialShape::dynamic());
    auto ptrue = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto pfalse = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto select = std::make_shared<op::v1::Select>(cond, ptrue, pfalse, op::AutoBroadcastType::NONE);

    std::vector<StaticShape> static_input_shapes = {StaticShape{6, 4}, StaticShape{6, 4}, StaticShape{6, 4}};
    const auto static_output_shapes = shape_inference(select.get(), static_input_shapes);
    ASSERT_EQ(static_output_shapes[0], StaticShape({6, 4}));
}
