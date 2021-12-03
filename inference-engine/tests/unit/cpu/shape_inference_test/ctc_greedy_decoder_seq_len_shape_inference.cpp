// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/op/parameter.hpp>
#include <openvino/op/ops.hpp>
#include <utils/shape_inference/shape_inference.hpp>
#include <utils/shape_inference/static_shape.hpp>

using namespace ov;

TEST(StaticShapeInferenceTest, CtcGreedyDecoderSeqLenTest) {
    auto P = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1});
    auto I = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1});
    auto G = std::make_shared<op::v6::CTCGreedyDecoderSeqLen>(P, I);

    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 100, 1200}, StaticShape{3}},
                             static_output_shapes = {StaticShape{}, StaticShape{}};
    shape_inference(G.get(), static_input_shapes, static_output_shapes);
    ASSERT_EQ(static_output_shapes[0], StaticShape({3, 100}));
    ASSERT_EQ(static_output_shapes[1], StaticShape({3}));
}