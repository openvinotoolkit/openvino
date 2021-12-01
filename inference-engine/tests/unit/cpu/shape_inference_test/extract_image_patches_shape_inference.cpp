// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <extract_image_patches_shape_inference.hpp>
#include <openvino/op/parameter.hpp>

#include "utils/shape_inference/static_shape.hpp"

using namespace ov;

TEST(StaticShapeInferenceTest, ExtractImagePatches) {
    auto data = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1, -1, -1, -1});
    auto sizes = Shape{3, 3};
    auto strides = Strides{5, 5};
    auto rates = Shape{1, 1};
    auto padTypePadding = op::PadType::VALID;
    auto extractImagePatches =
        std::make_shared<op::v3::ExtractImagePatches>(data, sizes, strides, rates, padTypePadding);

    std::vector<PartialShape> input_shapes = {PartialShape{64, 3, 10, 10}}, output_shapes = {PartialShape{}};

    shape_infer(extractImagePatches.get(), input_shapes, output_shapes);
    ASSERT_EQ(output_shapes[0], (PartialShape{64, 27, 2, 2}));

    std::vector<StaticShape> static_input_shapes = {StaticShape{64, 3, 10, 10}}, static_output_shapes = {StaticShape{}};

    shape_infer(extractImagePatches.get(), static_input_shapes, static_output_shapes);
    ASSERT_EQ(output_shapes[0], (PartialShape{64, 27, 2, 2}));
}
