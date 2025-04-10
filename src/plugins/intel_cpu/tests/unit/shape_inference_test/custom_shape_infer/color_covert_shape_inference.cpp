// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/op/ops.hpp"
#include "custom_shape_infer.hpp"
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

} // namespace cpu_shape_infer
} // namespace unit_test
} // namespace intel_cpu
} // namespace ov

