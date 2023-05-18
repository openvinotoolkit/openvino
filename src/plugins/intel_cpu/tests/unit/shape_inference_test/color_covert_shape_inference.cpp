// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;

TEST(StaticShapeInferenceTest, ColorConvertNV12toBGR) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto op = std::make_shared<op::v8::NV12toBGR>(data);
    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 720, 640, 1}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{}};
    shape_inference(op.get(), static_input_shapes, static_output_shapes);
    ASSERT_EQ(static_output_shapes[0], StaticShape({1, 480, 640, 3}));
    unit_test::cus_usual_shape_infer(op.get(), static_input_shapes, static_output_shapes);
}

TEST(StaticShapeInferenceTest, ColorConvertNV12toBGRMutliPlane) {
    auto dataY = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto dataUV = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto op = std::make_shared<op::v8::NV12toBGR>(dataY, dataUV);
    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 480, 640, 1}, StaticShape{1, 240, 320, 2}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{}};
    shape_inference(op.get(), static_input_shapes, static_output_shapes);
    ASSERT_EQ(static_output_shapes[0], StaticShape({1, 480, 640, 3}));
    unit_test::cus_usual_shape_infer(op.get(), static_input_shapes, static_output_shapes);
}

TEST(StaticShapeInferenceTest, ColorConvertNV12toRGB) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto op = std::make_shared<op::v8::NV12toRGB>(data);
    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 720, 640, 1}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{}};
    shape_inference(op.get(), static_input_shapes, static_output_shapes);
    ASSERT_EQ(static_output_shapes[0], StaticShape({1, 480, 640, 3}));
    unit_test::cus_usual_shape_infer(op.get(), static_input_shapes, static_output_shapes);
}

TEST(StaticShapeInferenceTest, ColorConvertNV12toRGBMutliPlane) {
    auto dataY = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto dataUV = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto op = std::make_shared<op::v8::NV12toRGB>(dataY, dataUV);
    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 480, 640, 1}, StaticShape{1, 240, 320, 2}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{}};
    shape_inference(op.get(), static_input_shapes, static_output_shapes);
    ASSERT_EQ(static_output_shapes[0], StaticShape({1, 480, 640, 3}));
    unit_test::cus_usual_shape_infer(op.get(), static_input_shapes, static_output_shapes);
}

TEST(StaticShapeInferenceTest, ColorConvertI420toBGR) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto op = std::make_shared<op::v8::I420toBGR>(data);
    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 720, 640, 1}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{}};
    shape_inference(op.get(), static_input_shapes, static_output_shapes);
    ASSERT_EQ(static_output_shapes[0], StaticShape({1, 480, 640, 3}));
    unit_test::cus_usual_shape_infer(op.get(), static_input_shapes, static_output_shapes);
}

TEST(StaticShapeInferenceTest, ColorConvertI420toBGRMutliPlane) {
    auto dataY = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto dataU = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto dataV = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto op = std::make_shared<op::v8::I420toBGR>(dataY, dataU, dataV);
    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 480, 640, 1}, StaticShape{1, 240, 320, 1}, StaticShape{1, 240, 320, 1}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{}};
    shape_inference(op.get(), static_input_shapes, static_output_shapes);
    ASSERT_EQ(static_output_shapes[0], StaticShape({1, 480, 640, 3}));
    unit_test::cus_usual_shape_infer(op.get(), static_input_shapes, static_output_shapes);
}

TEST(StaticShapeInferenceTest, ColorConvertI420toRGB) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto op = std::make_shared<op::v8::I420toRGB>(data);
    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 720, 640, 1}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{}};
    shape_inference(op.get(), static_input_shapes, static_output_shapes);
    ASSERT_EQ(static_output_shapes[0], StaticShape({1, 480, 640, 3}));
    unit_test::cus_usual_shape_infer(op.get(), static_input_shapes, static_output_shapes);
}

TEST(StaticShapeInferenceTest, ColorConvertI420toRGBMutliPlane) {
    auto dataY = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto dataU = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto dataV = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto op = std::make_shared<op::v8::I420toRGB>(dataY, dataU, dataV);
    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 480, 640, 1}, StaticShape{1, 240, 320, 1}, StaticShape{1, 240, 320, 1}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{}};
    shape_inference(op.get(), static_input_shapes, static_output_shapes);
    ASSERT_EQ(static_output_shapes[0], StaticShape({1, 480, 640, 3}));
    unit_test::cus_usual_shape_infer(op.get(), static_input_shapes, static_output_shapes);
}

TEST(StaticCustomShapeInferenceTest, novalid_input) {
    auto dataY = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto dataU = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto dataV = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto op = std::make_shared<op::v8::I420toRGB>(dataY, dataU, dataV);
    std::vector<StaticShape> static_input_shapes = {StaticShape{480, 640, 1}, StaticShape{240, 320, 1}, StaticShape{240, 320, 1}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{}};

    OV_EXPECT_THROW(unit_test::cus_usual_shape_infer(op.get(), static_input_shapes, static_output_shapes),
                    InferenceEngine::GeneralError,
                    testing::HasSubstr("NV12Converter node has incorrect input dimensions"));
}


