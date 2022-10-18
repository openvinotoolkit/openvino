// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <gather_shape_inference.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/ops.hpp>
#include <openvino/op/parameter.hpp>
#include <utils/shape_inference/shape_inference.hpp>
#include <utils/shape_inference/static_shape.hpp>

using namespace ov;
using namespace ov::intel_cpu;

TEST(StaticShapeInferenceTest, GatherV1Test) {
    auto P = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1});
    auto I = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1, -1});
    auto A = op::v0::Constant::create(element::i64, Shape{}, {0});
    auto G = std::make_shared<op::v1::Gather>(P, I, A);
    // Test StaticShape
    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 2}, StaticShape{2, 2}, StaticShape{1}},
                             static_output_shapes = {StaticShape{}};
    shape_inference(G.get(), static_input_shapes, static_output_shapes);
    ASSERT_EQ(static_output_shapes[0], (StaticShape{2, 2, 2}));
}

TEST(StaticShapeInferenceTest, GatherV1TestNonConstantA) {
    auto P = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1});
    auto I = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1, -1});
    auto A = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{});
    auto G = std::make_shared<op::v1::Gather>(P, I, A);
    auto hostTensor = std::make_shared<HostTensor>(element::i32, Shape{});
    int32_t val_a = 1;
    hostTensor->write(&val_a, sizeof(int32_t));
    // Test StaticShape
    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 2}, StaticShape{2, 2}, StaticShape{}},
                             static_output_shapes = {StaticShape{}};
    shape_inference(G.get(), static_input_shapes, static_output_shapes, {{2, hostTensor}});
    ASSERT_EQ(static_output_shapes[0], (StaticShape{3, 2, 2}));
}

TEST(StaticShapeInferenceTest, GatherV7Test) {
    auto P = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1});
    auto I = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1, -1});
    auto A = op::v0::Constant::create(element::i64, Shape{}, {0});
    auto G = std::make_shared<op::v7::Gather>(P, I, A);
    // Test StaticShape
    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 2}, StaticShape{2, 2}, StaticShape{1}},
                             static_output_shapes = {StaticShape{}};
    shape_inference(G.get(), static_input_shapes, static_output_shapes);
    ASSERT_EQ(static_output_shapes[0], (StaticShape{2, 2, 2}));
}

TEST(StaticShapeInferenceTest, GatherV7TestNonConstantA) {
    auto P = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1});
    auto I = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1, -1});
    auto A = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{});
    auto G = std::make_shared<op::v7::Gather>(P, I, A);
    auto hostTensor = std::make_shared<HostTensor>(element::i32, Shape{});
    int32_t val_a = 0;
    hostTensor->write(&val_a, sizeof(int32_t));
    // Test StaticShape
    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 2}, StaticShape{2, 2}, StaticShape{}},
                             static_output_shapes = {StaticShape{}};
    shape_inference(G.get(), static_input_shapes, static_output_shapes, {{2, hostTensor}});
    ASSERT_EQ(static_output_shapes[0], (StaticShape{2, 2, 2}));
}

TEST(StaticShapeInferenceTest, GatherV8Test) {
    auto P = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1});
    auto I = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1, -1});
    auto A = op::v0::Constant::create(element::i64, Shape{}, {0});
    auto G = std::make_shared<op::v8::Gather>(P, I, A);
    // Test StaticShape
    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 2}, StaticShape{2, 2}, StaticShape{1}},
                             static_output_shapes = {StaticShape{}};
    shape_inference(G.get(), static_input_shapes, static_output_shapes);
    ASSERT_EQ(static_output_shapes[0], (StaticShape{2, 2, 2}));
}

TEST(StaticShapeInferenceTest, GatherV8TestNonConstantA) {
    auto P = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1});
    auto I = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1, -1});
    auto A = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{});
    auto G = std::make_shared<op::v8::Gather>(P, I, A);
    auto hostTensor = std::make_shared<HostTensor>(element::i32, Shape{});
    int32_t val_a = 0;
    hostTensor->write(&val_a, sizeof(int32_t));
    // Test StaticShape
    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 2}, StaticShape{2, 2}, StaticShape{}},
                             static_output_shapes = {StaticShape{}};
    shape_inference(G.get(), static_input_shapes, static_output_shapes, {{2, hostTensor}});
    ASSERT_EQ(static_output_shapes[0], (StaticShape{2, 2, 2}));
}