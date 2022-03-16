// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/op/parameter.hpp>
#include <openvino/op/space_to_depth.hpp>
#include <utils/shape_inference/shape_inference.hpp>
#include <utils/shape_inference/static_shape.hpp>

using namespace ov::intel_cpu;

TEST(StaticShapeInferenceTest, SpaceToDepthTest) {
    auto A = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(ov::Rank(4)));
    auto space_to_depth =
        std::make_shared<ov::op::v0::SpaceToDepth>(A, ov::op::v0::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST, 2);
    const std::vector<StaticShape> input_shapes = {StaticShape{1, 12, 4, 1080, 1616}};
    std::vector<StaticShape> output_shapes = {StaticShape{}};
    shape_inference(space_to_depth.get(), input_shapes, output_shapes);
    ASSERT_EQ(output_shapes[0], (StaticShape{1, 12 * 8, 4 / 2, 1080 / 2, 1616 / 2}));
}
