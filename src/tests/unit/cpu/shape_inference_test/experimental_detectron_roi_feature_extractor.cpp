// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <convolution_shape_inference.hpp>
#include <experimental_detectron_roi_feature_shape_inference.hpp>
#include <openvino/op/ops.hpp>
#include <openvino/op/parameter.hpp>

#include "utils/shape_inference/shape_inference.hpp"
#include "utils/shape_inference/static_shape.hpp"

using namespace ov;
using namespace ov::intel_cpu;

TEST(StaticShapeInferenceTest, ExperimentalDetectronROIFeatureExtractor) {
    op::v6::ExperimentalDetectronROIFeatureExtractor::Attributes attrs;
    attrs.aligned = false;
    attrs.output_size = 14;
    attrs.sampling_ratio = 2;
    attrs.pyramid_scales = {4, 8, 16, 32};

    auto input = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1});
    auto pyramid_layer0 = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, -1, -1, -1});
    auto pyramid_layer1 = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, -1, -1, -1});
    auto pyramid_layer2 = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, -1, -1, -1});
    auto pyramid_layer3 = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, -1, -1, -1});

    auto roi = std::make_shared<op::v6::ExperimentalDetectronROIFeatureExtractor>(
        NodeVector{input, pyramid_layer0, pyramid_layer1, pyramid_layer2, pyramid_layer3},
        attrs);

    std::vector<StaticShape> input_shapes = {StaticShape{1000, 4},
                                             StaticShape{1, 256, 200, 336},
                                             StaticShape{1, 256, 100, 168},
                                             StaticShape{1, 256, 50, 84},
                                             StaticShape{1, 256, 25, 42}};
    std::vector<StaticShape> output_shapes = {StaticShape{}, StaticShape{}};
    shape_inference(roi.get(), input_shapes, output_shapes);

    EXPECT_EQ(output_shapes[0], (StaticShape{1000, 256, 14, 14}));
    EXPECT_EQ(output_shapes[1], (StaticShape{1000, 4}));
}
