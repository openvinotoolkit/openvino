// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <extract_image_patches_shape_inference.hpp>
#include <openvino/op/ops.hpp>
#include <openvino/op/parameter.hpp>
#include <utils/shape_inference/shape_inference.hpp>
#include <utils/shape_inference/static_shape.hpp>

using namespace ov;
using namespace ov::intel_cpu;

TEST(StaticShapeInferenceTest, ExtractImagePatchesTest) {
    auto data = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1, -1, -1, -1});
    auto sizes = Shape{3, 3};
    auto strides = Strides{5, 5};
    auto rates = Shape{1, 1};
    auto padTypePadding = op::PadType::VALID;
    auto extractImagePatches =
        std::make_shared<op::v3::ExtractImagePatches>(data, sizes, strides, rates, padTypePadding);
    // Test StaticShape
    std::vector<StaticShape> static_input_shapes = {StaticShape{64, 3, 10, 10}}, static_output_shapes = {StaticShape{}};

    shape_inference(extractImagePatches.get(), static_input_shapes, static_output_shapes);
    ASSERT_EQ(static_output_shapes[0], (StaticShape{64, 27, 2, 2}));
}
