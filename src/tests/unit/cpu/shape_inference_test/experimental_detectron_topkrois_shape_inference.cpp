// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <experimental_detectron_topkrois_shape_inference.hpp>
#include <openvino/core/coordinate_diff.hpp>
#include <openvino/op/ops.hpp>
#include <openvino/op/parameter.hpp>

#include "utils/shape_inference/static_shape.hpp"

using namespace ov;
using namespace ov::intel_cpu;

TEST(StaticShapeInferenceTest, ExperimentalDetectronTopKROIsTest) {
    auto input_rois = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1});
    auto rois_probs = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1});
    size_t max_rois = 5;

    auto rois = std::make_shared<op::v6::ExperimentalDetectronTopKROIs>(input_rois, rois_probs, max_rois);

    std::vector<PartialShape> input_shapes = {PartialShape{10, 4}, PartialShape{10}}, output_shapes = {PartialShape{}};
    shape_infer(rois.get(), input_shapes, output_shapes);

    ASSERT_EQ(output_shapes[0], PartialShape({5, 4}));

    std::vector<StaticShape> static_input_shapes = {StaticShape{10, 4}, StaticShape{10}}, static_output_shapes = {StaticShape{}};
    shape_infer(rois.get(), static_input_shapes, static_output_shapes);

    ASSERT_EQ(static_output_shapes[0], StaticShape({5, 4}));
}
