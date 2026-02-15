// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/convolution.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "custom_shape_infer.hpp"
#include "openvino/op/parameter.hpp"

namespace ov {
namespace intel_cpu {
namespace unit_test {
namespace cpu_shape_infer {

using namespace ov;
using namespace ov::intel_cpu;

TEST(CpuShapeInfer, Conv2D_DefaultCtor) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto filters = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto op = std::make_shared<ov::op::v1::Convolution>(data,
                                                        filters,
                                                        Strides{1, 1},
                                                        CoordinateDiff{2, 2},
                                                        CoordinateDiff{2, 1},
                                                        Strides{1, 1},
                                                        op::PadType::VALID);
    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 3, 10, 12}, {2, 3, 5, 5}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{1, 2, 6, 8}};
    unit_test::cpu_test_shape_infer(op.get(), static_input_shapes, static_output_shapes);
}

TEST(CpuShapeInfer, Conv2D_ThreeInputShapes) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto filters = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto op = std::make_shared<ov::op::v1::Convolution>(data,
                                                        filters,
                                                        Strides{1, 1},
                                                        CoordinateDiff{2, 2},
                                                        CoordinateDiff{2, 1},
                                                        Strides{1, 1},
                                                        op::PadType::VALID);
    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 3, 10, 12}, {2, 3, 5, 5}, {2}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{1, 2, 6, 8}};
    unit_test::cpu_test_shape_infer(op.get(), static_input_shapes, static_output_shapes);
}

TEST(CpuShapeInfer, Conv2D_AutoPadsSameLower) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto filters = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto op = std::make_shared<ov::op::v1::Convolution>(data,
                                                        filters,
                                                        Strides{1, 1},
                                                        CoordinateDiff{0, 0},
                                                        CoordinateDiff{0, 0},
                                                        Strides{1, 1},
                                                        op::PadType::SAME_LOWER);
    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 6, 5, 5}, {7, 6, 3, 3}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{3, 7, 5, 5}};
    unit_test::cpu_test_shape_infer(op.get(), static_input_shapes, static_output_shapes);
}

TEST(CpuShapeInfer, Conv3D_AutoPadsSameUpper) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1, -1});
    auto filters = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1, -1});
    auto op = std::make_shared<ov::op::v1::Convolution>(data,
                                                        filters,
                                                        Strides{1, 1, 1},
                                                        CoordinateDiff{0, 0, 0},
                                                        CoordinateDiff{0, 0, 0},
                                                        Strides{1, 1, 1},
                                                        op::PadType::SAME_UPPER);
    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 6, 5, 5, 5}, {7, 6, 3, 3, 3}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{3, 7, 5, 5, 5}};
    unit_test::cpu_test_shape_infer(op.get(), static_input_shapes, static_output_shapes);
}

TEST(CpuShapeInfer, Conv3D_DataAndFiltersNumChannelsMismatch) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1, -1});
    auto filters = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1, -1});
    auto op = std::make_shared<ov::op::v1::Convolution>(data,
                                                        filters,
                                                        Strides{1, 1, 1},
                                                        CoordinateDiff{0, 0, 0},
                                                        CoordinateDiff{0, 0, 0},
                                                        Strides{1, 1, 1},
                                                        op::PadType::SAME_UPPER);
    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 5, 5, 5, 5}, {7, 6, 3, 3, 3}};
    std::vector<StaticShape> static_output_shapes = {};

    OV_EXPECT_THROW(unit_test::cpu_test_shape_infer(op.get(), static_input_shapes, static_output_shapes),
                    ov::Exception,
                    testing::HasSubstr("Input and filter channels must match"));
}

}  // namespace cpu_shape_infer
}  // namespace unit_test
}  // namespace intel_cpu
}  // namespace ov
