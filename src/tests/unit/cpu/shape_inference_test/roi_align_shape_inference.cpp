// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/op/parameter.hpp>
#include <openvino/op/roi_align.hpp>
#include <utils/shape_inference/shape_inference.hpp>
#include <utils/shape_inference/static_shape.hpp>

using namespace ov;

TEST(StaticShapeInferenceTest, ROIAlignTest) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    const auto rois = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1});
    const auto batch_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto op = std::make_shared<op::v3::ROIAlign>(data, rois, batch_indices, 2, 2, 1, 1.0f, "avg");
    const std::vector<ov::StaticShape> input_shapes = {ov::StaticShape{2, 3, 5, 5},
                                                       ov::StaticShape{7, 4},
                                                       ov::StaticShape{7}};
    std::vector<ov::StaticShape> output_shapes = {ov::StaticShape{}};
    shape_inference(op.get(), input_shapes, output_shapes);

    ASSERT_EQ(output_shapes[0], (StaticShape{7, 3, 2, 2}));
}