// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/core/coordinate_diff.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/relu.hpp>
#include <openvino/op/fake_quantize.hpp>
#include <utils/shape_inference/shape_inference.hpp>
#include <utils/shape_inference/static_shape.hpp>

using namespace ov;
using namespace ov::intel_cpu;

TEST(StaticShapeInferenceTest, UnaryEltwiseTest) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto node = std::make_shared<op::v0::Relu>(data);

    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 6, 5, 5}},
        static_output_shapes = {StaticShape{}};
    shape_inference(node.get(), static_input_shapes, static_output_shapes);

    ASSERT_EQ(static_output_shapes[0], StaticShape({3, 6, 5, 5}));
}

TEST(StaticShapeInferenceTest, BinaryEltwiseTest) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto data_1 = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1});

    auto node = std::make_shared<op::v1::Add>(data, data_1);

    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 6, 1, 5}, StaticShape{1, 3, 5}},
            static_output_shapes = {StaticShape{}};
    shape_inference(node.get(), static_input_shapes, static_output_shapes);

    ASSERT_EQ(static_output_shapes[0], StaticShape({3, 6, 3, 5}));
}

TEST(StaticShapeInferenceTest, FakeQuantizeTest) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto il = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1});
    auto ih = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1});
    auto ol = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1});
    auto oh = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1});

    auto node = std::make_shared<op::v0::FakeQuantize>(data, il, ih, ol, oh, 256);

    std::vector<StaticShape> static_input_shapes = {
                StaticShape{3, 6, 3, 5},
                StaticShape{1, 3, 1},
                StaticShape{1},
                StaticShape{5},
                StaticShape{1, 1, 1, 1}
            },
            static_output_shapes = {StaticShape{}};

    shape_inference(node.get(), static_input_shapes, static_output_shapes);
    ASSERT_EQ(static_output_shapes[0], StaticShape({3, 6, 3, 5}));
}
