// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "utils.hpp"
#include "openvino/opsets/opset9.hpp"
#include "grid_sample_shape_inference.hpp"

using namespace ov;
using namespace ov::intel_cpu;

TEST(StaticShapeInferenceTest, GridSample) {
    const auto data = std::make_shared<opset9::Parameter>(element::i32, PartialShape{-1, -1, -1, -1});
    const auto grid = std::make_shared<opset9::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});

    const auto grid_sample = std::make_shared<opset9::GridSample>(data, grid, opset9::GridSample::Attributes{});

    std::vector<StaticShape> static_input_shapes = {StaticShape{2, 3, 4, 8}, StaticShape{2, 6, 7, 2}},
                             static_output_shapes = {StaticShape{}};
    shape_inference(grid_sample.get(), static_input_shapes, static_output_shapes);
    EXPECT_EQ(static_output_shapes[0], StaticShape({2, 3, 6, 7}));
}

TEST(StaticShapeInferenceTest, GridSampleDefaultConstructor) {
    auto grid_sample = std::make_shared<opset9::GridSample>();

    std::vector<StaticShape> static_input_shapes = {StaticShape{2, 3, 4, 8}, StaticShape{2, 6, 7, 2}},
                             static_output_shapes = {StaticShape{}};
    shape_infer(grid_sample.get(), static_input_shapes, static_output_shapes);
    EXPECT_EQ(static_output_shapes[0], StaticShape({2, 3, 6, 7}));
}
