// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/op/depth_to_space.hpp>
#include <openvino/op/parameter.hpp>
#include <utils/shape_inference/shape_inference.hpp>
#include <utils/shape_inference/static_shape.hpp>

using namespace ov::intel_cpu;

TEST(StaticShapeInferenceTest, DepthToSpaceTest) {
    auto A = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(ov::Rank(4)));
    auto depth_to_space =
        std::make_shared<ov::op::v0::DepthToSpace>(A, ov::op::v0::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST, 2);
    const std::vector<StaticShape> input_shapes = {StaticShape{1, 16, 3, 1080, 1616}};
    std::vector<StaticShape> output_shapes = {StaticShape{}};
    shape_inference(depth_to_space.get(), input_shapes, output_shapes);
    ASSERT_EQ(output_shapes[0], (StaticShape{1, 2, 2 * 3, 2 * 1080, 2 * 1616}));
}
