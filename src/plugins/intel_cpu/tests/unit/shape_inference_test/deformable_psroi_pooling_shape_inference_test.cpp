// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gmock/gmock.h>

#include <array>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/opsets/opset10.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace ov::opset10;
using namespace testing;

class DeformablePSROIPoolingV1StaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v1::DeformablePSROIPooling> {
protected:
    void SetUp() override {
        output_shapes.resize(1);
    }
};

TEST_F(DeformablePSROIPoolingV1StaticShapeInferenceTest, default_ctor) {
    const auto op = make_op();

    const int64_t output_dim = 88;
    const int64_t group_size = 2;

    const auto rois_dim = 30;

    op->set_output_dim(output_dim);
    op->set_group_size(group_size);

    auto expected_output = StaticShape{rois_dim, output_dim, group_size, group_size};

    // 2 inputs
    {
        input_shapes = {StaticShape{2, 4, 8, 6}, StaticShape{rois_dim, 5}};
        output_shapes = shape_inference(op.get(), input_shapes);
        EXPECT_EQ(output_shapes[0], expected_output);
    }
    // 3 inputs
    {
        input_shapes = {StaticShape{2, 4, 8, 6}, StaticShape{rois_dim, 5}, StaticShape{rois_dim, 20, group_size, group_size}};
        output_shapes = shape_inference(op.get(), input_shapes);
        EXPECT_EQ(output_shapes[0], expected_output);
    }
}

TEST_F(DeformablePSROIPoolingV1StaticShapeInferenceTest, no_offsets_input) {
    const float spatial_scale = 0.05f;
    const int64_t output_dim = 88;
    const int64_t group_size = 2;

    const auto rois_dim = 30;

    auto input_data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    auto input_coords = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());

    auto op = make_op(input_data, input_coords, output_dim, spatial_scale, group_size);

    StaticShape expected_output{rois_dim, output_dim, group_size, group_size};
    input_shapes = {StaticShape{2, 4, 8, 6}, StaticShape{rois_dim, 5}};

    output_shapes = shape_inference(op.get(), input_shapes);
    EXPECT_EQ(output_shapes[0], expected_output);
}

TEST_F(DeformablePSROIPoolingV1StaticShapeInferenceTest, offsets_input) {
    const float spatial_scale = 0.05f;
    const int64_t output_dim = 88;
    const int64_t group_size = 2;

    const auto rois_dim = 30;

    auto input_data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    auto input_coords = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    auto input_offsets = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());

    auto op = make_op(input_data, input_coords, input_offsets, output_dim, spatial_scale, group_size);

    StaticShape expected_output{rois_dim, output_dim, group_size, group_size};
    input_shapes = {StaticShape{2, 4, 8, 6}, StaticShape{rois_dim, 5}, StaticShape{rois_dim, 20, group_size, group_size}};

    output_shapes = shape_inference(op.get(), input_shapes);
    EXPECT_EQ(output_shapes[0], expected_output);
}
