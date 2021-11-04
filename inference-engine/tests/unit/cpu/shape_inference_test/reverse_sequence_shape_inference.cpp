// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <reverse_sequence_shape_inference.hpp>
#include <openvino/core/coordinate_diff.hpp>
#include <openvino/op/ops.hpp>
#include <openvino/op/parameter.hpp>

#include "utils/shape_inference/static_shape.hpp"

using namespace ov;

TEST(StaticShapeInferenceTest, ReverseSequenceTest) {
    auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1});
    auto seq_lengths = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{4});
    auto reverse_seq = std::make_shared<op::v0::ReverseSequence>(data, seq_lengths);
    std::vector<PartialShape> input_shapes = {PartialShape{4, 3, 2}, PartialShape{4}},
                              output_shapes = {PartialShape{}};
    shape_infer(reverse_seq.get(), input_shapes, output_shapes);
    ASSERT_EQ(output_shapes[0], (PartialShape{4, 3, 2}));
}