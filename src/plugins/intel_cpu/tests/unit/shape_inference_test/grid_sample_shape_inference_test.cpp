// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "utils.hpp"
#include "openvino/opsets/opset9.hpp"
#include "grid_sample_shape_inference.hpp"

using namespace ov;
using namespace ov::intel_cpu;

class GridSampleStaticShapeInferenceTest : public OpStaticShapeInferenceTest<ov::op::v9::GridSample> {};

TEST_F(GridSampleStaticShapeInferenceTest, GridSample) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1, -1, -1, -1});
    const auto grid = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});

    op = make_op(data, grid, opset9::GridSample::Attributes{});

    input_shapes = {StaticShape{2, 3, 4, 8}, StaticShape{2, 6, 7, 2}};
    output_shapes = {StaticShape{}};
    exp_shape = StaticShape{2, 3, 6, 7};

    output_shapes = shape_inference(op.get(), input_shapes);
    EXPECT_EQ(output_shapes[0], exp_shape);
}

TEST_F(GridSampleStaticShapeInferenceTest, GridSample_default_constructor) {
    op = make_op();

    input_shapes = {StaticShape{2, 3, 4, 8}, StaticShape{2, 6, 7, 2}};
    output_shapes = {StaticShape{}};
    exp_shape = StaticShape{2, 3, 6, 7};

    output_shapes = shape_inference(op.get(), input_shapes);
    EXPECT_EQ(output_shapes[0], exp_shape);
}
