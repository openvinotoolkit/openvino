// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <bucketize_shape_inference.hpp>
#include <openvino/core/coordinate_diff.hpp>
#include <openvino/op/convolution.hpp>
#include <openvino/op/ops.hpp>
#include <openvino/op/parameter.hpp>

#include "utils/shape_inference/static_shape.hpp"

using namespace ov;
using namespace std;

TEST(StaticShapeInferenceTest, BucketizeV3) {
    auto data = make_shared<op::v0::Parameter>(element::f32, ov::Shape{2, 3, 2});
    auto buckets = make_shared<op::v0::Parameter>(element::f32, ov::Shape{4});
    auto bucketize = make_shared<op::v3::Bucketize>(data, buckets);

    EXPECT_TRUE(bucketize->get_output_partial_shape(0).same_scheme(PartialShape{2, 3, 2}));

    std::vector<PartialShape> input_shapes = {PartialShape{2, 3, 2}, ov::Shape{4}};
    std::vector<PartialShape> output_shapes = {PartialShape::dynamic()};
    shape_infer(bucketize.get(), input_shapes, output_shapes);
    ASSERT_EQ(output_shapes.size(), 1);
    ASSERT_EQ(output_shapes[0], PartialShape({2, 3, 2}));

    std::vector<StaticShape> static_input_shapes = {StaticShape{2, 3, 2}, StaticShape{4}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{}};
    shape_infer(bucketize.get(), static_input_shapes, static_output_shapes);

    ASSERT_EQ(static_output_shapes.size(), 1);
    ASSERT_EQ(static_output_shapes[0], StaticShape({2, 3, 2}));
}
