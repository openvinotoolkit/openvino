// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "custom_shape_infer.hpp"
#include "openvino/op/bgr_to_nv12.hpp"
#include "openvino/op/i420_to_bgr.hpp"
#include "openvino/op/i420_to_rgb.hpp"
#include "openvino/op/nv12_to_bgr.hpp"
#include "openvino/op/nv12_to_rgb.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/rgb_to_nv12.hpp"
namespace ov {
namespace intel_cpu {
namespace unit_test {
namespace cpu_shape_infer {

using namespace ov;
using namespace ov::intel_cpu;

template <class T>
class CpuShapeInferenceTest_ColorConvertNV12 : public testing::Test {};

// CpuShapeInferenceTest for BinaryElementwiseArithmetis (ColorConvert) operations
TYPED_TEST_SUITE_P(CpuShapeInferenceTest_ColorConvertNV12);

TYPED_TEST_P(CpuShapeInferenceTest_ColorConvertNV12, singlePlane) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto op = std::make_shared<TypeParam>(data);
    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 720, 640, 1}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{1, 480, 640, 3}};
    unit_test::cpu_test_shape_infer(op.get(), static_input_shapes, static_output_shapes);
}

TYPED_TEST_P(CpuShapeInferenceTest_ColorConvertNV12, multiPlane) {
    auto dataY = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto dataUV = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto op = std::make_shared<TypeParam>(dataY, dataUV);
    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 480, 640, 1}, StaticShape{1, 240, 320, 2}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{1, 480, 640, 3}};
    unit_test::cpu_test_shape_infer(op.get(), static_input_shapes, static_output_shapes);
}

TYPED_TEST_P(CpuShapeInferenceTest_ColorConvertNV12, novalid_input) {
    auto dataY = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto dataUV = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto op = std::make_shared<TypeParam>(dataY, dataUV);
    std::vector<StaticShape> static_input_shapes = {StaticShape{480, 640, 1}, StaticShape{240, 320, 2}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{}};

    OV_EXPECT_THROW(unit_test::cpu_test_shape_infer(op.get(), static_input_shapes, static_output_shapes),
                    ov::Exception,
                    testing::HasSubstr("NV12Converter node has incorrect input dimensions"));
}

REGISTER_TYPED_TEST_SUITE_P(CpuShapeInferenceTest_ColorConvertNV12,
                            singlePlane,
                            multiPlane,
                            novalid_input);

INSTANTIATE_TYPED_TEST_SUITE_P(CpuShapeInfer_NV12toBGR, CpuShapeInferenceTest_ColorConvertNV12, ::testing::Types<op::v8::NV12toBGR>);
INSTANTIATE_TYPED_TEST_SUITE_P(CpuShapeInfer_NV12toRGB, CpuShapeInferenceTest_ColorConvertNV12, ::testing::Types<op::v8::NV12toRGB>);

template <class T>
class CpuShapeInferenceTest_ColorConvertI420 : public testing::Test {};

// CpuShapeInferenceTest for BinaryElementwiseArithmetis (ColorConvert) operations
TYPED_TEST_SUITE_P(CpuShapeInferenceTest_ColorConvertI420);

TYPED_TEST_P(CpuShapeInferenceTest_ColorConvertI420, singlePlane) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto op = std::make_shared<TypeParam>(data);
    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 720, 640, 1}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{1, 480, 640, 3}};
    unit_test::cpu_test_shape_infer(op.get(), static_input_shapes, static_output_shapes);
}

TYPED_TEST_P(CpuShapeInferenceTest_ColorConvertI420, multiPlane) {
    auto dataY = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto dataU = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto dataV = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto op = std::make_shared<TypeParam>(dataY, dataU, dataV);
    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 480, 640, 1}, StaticShape{1, 240, 320, 1}, StaticShape{1, 240, 320, 1}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{1, 480, 640, 3}};
    unit_test::cpu_test_shape_infer(op.get(), static_input_shapes, static_output_shapes);
}

TYPED_TEST_P(CpuShapeInferenceTest_ColorConvertI420, novalid_input) {
    auto dataY = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto dataU = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto dataV = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto op = std::make_shared<TypeParam>(dataY, dataU, dataV);
    std::vector<StaticShape> static_input_shapes = {StaticShape{480, 640, 1}, StaticShape{240, 320, 1}, StaticShape{240, 320, 1}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{}};

    OV_EXPECT_THROW(unit_test::cpu_test_shape_infer(op.get(), static_input_shapes, static_output_shapes),
                    ov::Exception,
                    testing::HasSubstr("NV12Converter node has incorrect input dimensions"));
}

REGISTER_TYPED_TEST_SUITE_P(CpuShapeInferenceTest_ColorConvertI420,
                            singlePlane,
                            multiPlane,
                            novalid_input);

INSTANTIATE_TYPED_TEST_SUITE_P(CpuShapeInfer_I420toBGR, CpuShapeInferenceTest_ColorConvertI420, ::testing::Types<op::v8::I420toBGR>);
INSTANTIATE_TYPED_TEST_SUITE_P(CpuShapeInfer_I420toRGB, CpuShapeInferenceTest_ColorConvertI420, ::testing::Types<op::v8::I420toRGB>);

template <class T>
class CpuShapeInferenceTest_ColorConvertToNV12 : public testing::Test {};

TYPED_TEST_SUITE_P(CpuShapeInferenceTest_ColorConvertToNV12);

TYPED_TEST_P(CpuShapeInferenceTest_ColorConvertToNV12, singlePlane) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto op = std::make_shared<TypeParam>(data, true);
    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 480, 640, 3}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{1, 720, 640, 1}};
    unit_test::cpu_test_shape_infer(op.get(), static_input_shapes, static_output_shapes);
}

TYPED_TEST_P(CpuShapeInferenceTest_ColorConvertToNV12, multiPlane) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto op = std::make_shared<TypeParam>(data, false);
    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 480, 640, 3}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{1, 480, 640, 1}, StaticShape{1, 240, 320, 2}};
    unit_test::cpu_test_shape_infer(op.get(), static_input_shapes, static_output_shapes);
}

TYPED_TEST_P(CpuShapeInferenceTest_ColorConvertToNV12, invalid_rank) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto op = std::make_shared<TypeParam>(data, false);
    std::vector<StaticShape> static_input_shapes = {StaticShape{480, 640, 3}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{}, StaticShape{}};

    OV_EXPECT_THROW(unit_test::cpu_test_shape_infer(op.get(), static_input_shapes, static_output_shapes),
                    ov::Exception,
                    testing::HasSubstr("RGBtoNV12/BGRtoNV12 node expects 4 input dimensions"));
}

TYPED_TEST_P(CpuShapeInferenceTest_ColorConvertToNV12, invalid_channels) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto op = std::make_shared<TypeParam>(data, true);
    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 4, 4, 2}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{}};

    OV_EXPECT_THROW(unit_test::cpu_test_shape_infer(op.get(), static_input_shapes, static_output_shapes),
                    ov::Exception,
                    testing::HasSubstr("RGB/BGR input number of channels should be equal to 3"));
}

TYPED_TEST_P(CpuShapeInferenceTest_ColorConvertToNV12, odd_height) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto op = std::make_shared<TypeParam>(data, false);
    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 5, 4, 3}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{}, StaticShape{}};

    OV_EXPECT_THROW(unit_test::cpu_test_shape_infer(op.get(), static_input_shapes, static_output_shapes),
                    ov::Exception,
                    testing::HasSubstr("Image height must be even"));
}

TYPED_TEST_P(CpuShapeInferenceTest_ColorConvertToNV12, odd_width) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto op = std::make_shared<TypeParam>(data, false);
    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 4, 3, 3}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{}, StaticShape{}};

    OV_EXPECT_THROW(unit_test::cpu_test_shape_infer(op.get(), static_input_shapes, static_output_shapes),
                    ov::Exception,
                    testing::HasSubstr("Image width must be even"));
}

REGISTER_TYPED_TEST_SUITE_P(CpuShapeInferenceTest_ColorConvertToNV12,
                            singlePlane,
                            multiPlane,
                            invalid_rank,
                            invalid_channels,
                            odd_height,
                            odd_width);

INSTANTIATE_TYPED_TEST_SUITE_P(CpuShapeInfer_RGBtoNV12,
                               CpuShapeInferenceTest_ColorConvertToNV12,
                               ::testing::Types<op::v17::RGBtoNV12>);
INSTANTIATE_TYPED_TEST_SUITE_P(CpuShapeInfer_BGRtoNV12,
                               CpuShapeInferenceTest_ColorConvertToNV12,
                               ::testing::Types<op::v17::BGRtoNV12>);

} // namespace cpu_shape_infer
} // namespace unit_test
} // namespace intel_cpu
} // namespace ov
