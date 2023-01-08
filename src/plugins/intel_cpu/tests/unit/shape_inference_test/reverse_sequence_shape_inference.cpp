// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/op/parameter.hpp>
#include <reverse_sequence_shape_inference.hpp>
#include <utils/shape_inference/shape_inference.hpp>
#include <utils/shape_inference/static_shape.hpp>

using namespace ov;
using namespace ov::intel_cpu;

TEST(StaticShapeInferenceTest, ReverseSequenceTest) {
    auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1});
    auto seq_lengths = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{4});
    auto reverse_seq = std::make_shared<op::v0::ReverseSequence>(data, seq_lengths);
    // Test StaticShape
    std::vector<StaticShape> static_input_shapes = {StaticShape{4, 3, 2}, StaticShape{4}},
                             static_output_shapes = {StaticShape{}};
    shape_inference(reverse_seq.get(), static_input_shapes, static_output_shapes);
    ASSERT_EQ(static_output_shapes[0], (StaticShape{4, 3, 2}));
}