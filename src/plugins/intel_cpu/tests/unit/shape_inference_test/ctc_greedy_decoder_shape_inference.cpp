// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ctc_greedy_decoder_shape_inference.hpp>
#include <openvino/op/ops.hpp>
#include <openvino/op/parameter.hpp>
#include <utils/shape_inference/shape_inference.hpp>
#include <utils/shape_inference/static_shape.hpp>

using namespace ov;
using namespace ov::intel_cpu;

TEST(StaticShapeInferenceTest, CtcGreedyDecoderTest) {
    auto P = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1});
    auto I = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1, -1});
    auto G = std::make_shared<op::v0::CTCGreedyDecoder>(P, I, false);
    // Test StaticShape
    std::vector<StaticShape> static_input_shapes = {StaticShape{100, 3, 1200}, StaticShape{100, 3}},
                             static_output_shapes = {StaticShape{}};
    shape_inference(G.get(), static_input_shapes, static_output_shapes);
    ASSERT_EQ(static_output_shapes[0], StaticShape({3, 100, 1, 1}));
}