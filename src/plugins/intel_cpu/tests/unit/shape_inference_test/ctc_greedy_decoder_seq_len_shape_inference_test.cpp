// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "ctc_greedy_decoder_seq_len_shape_inference.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/op/ops.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

class CTCGreedyDecoderSeqLenV6StaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v6::CTCGreedyDecoderSeqLen> {
    void SetUp() override {
        output_shapes.resize(1);
    }
};

TEST_F(CTCGreedyDecoderSeqLenV6StaticShapeInferenceTest, basic) {
    auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1});
    auto seq_mask = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    op = make_op(data, seq_mask, false);

    input_shapes = {StaticShape{4, 100, 1200}, StaticShape{4}};

    output_shapes = shape_inference(op.get(), input_shapes);
    EXPECT_EQ(output_shapes[0], StaticShape({4, 100}));
    EXPECT_EQ(output_shapes[1], StaticShape({4}));
}

TEST_F(CTCGreedyDecoderSeqLenV6StaticShapeInferenceTest, default_ctor) {
    op = make_op();

    // Two inputs
    input_shapes = {StaticShape{4, 100, 1200}, StaticShape{4}};
    output_shapes = shape_inference(op.get(), input_shapes);
    EXPECT_EQ(output_shapes[0], StaticShape({4, 100}));
    EXPECT_EQ(output_shapes[1], StaticShape({4}));

    // Three inputs (the last one is optional)
    input_shapes = {StaticShape{4, 100, 1200}, StaticShape{4}, {}};
    output_shapes = shape_inference(op.get(), input_shapes);
    EXPECT_EQ(output_shapes[0], StaticShape({4, 100}));
    EXPECT_EQ(output_shapes[1], StaticShape({4}));
}

TEST_F(CTCGreedyDecoderSeqLenV6StaticShapeInferenceTest, incompatible_batch) {
    auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto seq_mask = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());
    op = make_op(data, seq_mask, false);

    input_shapes = {StaticShape{4, 100, 1200}, StaticShape{6}};

    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes),
                    NodeValidationFailure,
                    HasSubstr("The first dimensions of input tensors must match"))
}

TEST_F(CTCGreedyDecoderSeqLenV6StaticShapeInferenceTest, incompatible_seq_len_rank) {
    auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto seq_mask = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());
    op = make_op(data, seq_mask, false);

    input_shapes = {StaticShape{4, 100, 1200}, StaticShape{4, 1}};

    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes),
                    NodeValidationFailure,
                    HasSubstr("The rank of sequence len tensor must be equal to 1"))
}
