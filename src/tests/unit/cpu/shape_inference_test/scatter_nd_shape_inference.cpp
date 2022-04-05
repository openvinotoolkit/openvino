// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/op/ops.hpp>
#include <openvino/op/parameter.hpp>
#include <scatter_nd_base_shape_inference.hpp>
#include <utils/shape_inference/shape_inference.hpp>
#include <utils/shape_inference/static_shape.hpp>

using namespace ov;
using namespace ov::intel_cpu;

TEST(StaticShapeInferenceTest, ScatterNDUpdateTest) {
    auto data_shape = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1, -1, -1, -1});
    auto indices_shape = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1, -1, -1});
    auto updates_shape = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1, -1, -1});
    auto scatter_nd_update = std::make_shared<op::v3::ScatterNDUpdate>(data_shape, indices_shape, updates_shape);

    std::vector<StaticShape> input_shapes = {StaticShape{1000, 256, 10, 15},
                                             StaticShape{25, 125, 3},
                                             StaticShape{25, 125, 15}},
                             output_shapes = {StaticShape{}};
    shape_inference(scatter_nd_update.get(), input_shapes, output_shapes);

    ASSERT_EQ(output_shapes[0], StaticShape({1000, 256, 10, 15}));
}