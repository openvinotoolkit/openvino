// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "ctc_greedy_decoder_shape_inference.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/op/ops.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

class CTCGreedyDecoderStaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v0::CTCGreedyDecoder> {};

TEST_F(CTCGreedyDecoderStaticShapeInferenceTest, ctc_greedy_decoder) {
    auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1});
    auto seq_mask = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1, -1});
    op = make_op(data, seq_mask, false);

    input_shapes = {StaticShape{100, 3, 1200}, StaticShape{100, 3}};
    output_shapes = {StaticShape{}};

    shape_inference(op.get(), input_shapes, output_shapes);
    EXPECT_EQ(output_shapes[0], StaticShape({3, 100, 1, 1}));
}

TEST_F(CTCGreedyDecoderStaticShapeInferenceTest, ctc_greedy_decoder_default_ctor) {
    op = make_op();

    input_shapes = {StaticShape{100, 3, 1200}, StaticShape{100, 3}};
    output_shapes = {StaticShape{}};

    shape_infer(op.get(), input_shapes, output_shapes);
    EXPECT_EQ(output_shapes[0], StaticShape({3, 100, 1, 1}));
}

TEST_F(CTCGreedyDecoderStaticShapeInferenceTest, ctc_greedy_decoder_incompatible_batch) {
    auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto seq_mask = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());
    op = make_op(data, seq_mask, false);

    input_shapes = {StaticShape{10, 3, 1200}, StaticShape{100, 3}};
    output_shapes = {StaticShape{}};

    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes, output_shapes),
                    NodeValidationFailure,
                    HasSubstr("The first dimensions of input tensors must match"))
}

TEST_F(CTCGreedyDecoderStaticShapeInferenceTest, ctc_greedy_decoder_incompatible_t_dim) {
    auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto seq_mask = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());
    op = make_op(data, seq_mask, false);

    input_shapes = {StaticShape{100, 3, 1200}, StaticShape{100, 5}};
    output_shapes = {StaticShape{}};

    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes, output_shapes),
                    NodeValidationFailure,
                    HasSubstr("The second dimensions of input tensors must match"))
}
