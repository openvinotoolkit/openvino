// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/op/experimental_detectron_prior_grid_generator.hpp>
#include <openvino/op/ops.hpp>
#include <openvino/op/parameter.hpp>
#include <utils/shape_inference/shape_inference.hpp>
#include <utils/shape_inference/static_shape.hpp>

using namespace ov;
using namespace ov::intel_cpu;

TEST(StaticShapeInferenceTest, PriorGridGenerator) {
    op::v6::ExperimentalDetectronPriorGridGenerator::Attributes attrs;
    attrs.flatten = false;
    attrs.h = 0;
    attrs.w = 0;
    attrs.stride_x = 4.0f;
    attrs.stride_y = 4.0f;

    auto priors = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1});
    auto feature_map = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto im_data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});

    auto grid_gen =
        std::make_shared<ov::op::v6::ExperimentalDetectronPriorGridGenerator>(priors, feature_map, im_data, attrs);

    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 4},
                                                    StaticShape{1, 256, 200, 336},
                                                    StaticShape{1, 3, 800, 1344}},
                             static_output_shapes = {StaticShape{}};
    shape_inference(grid_gen.get(), static_input_shapes, static_output_shapes);

    ASSERT_EQ(static_output_shapes[0], StaticShape({200, 336, 3, 4}));
}