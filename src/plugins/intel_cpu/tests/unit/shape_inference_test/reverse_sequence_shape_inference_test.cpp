// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_assertions.hpp"
#include "openvino/opsets/opset10.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace ov::opset10;
using namespace testing;

class ReverseSequenceV0StaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v0::ReverseSequence> {
protected:
    void SetUp() override {
        output_shapes.resize(1);

        data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
        seq_lengths = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    }

    std::shared_ptr<Parameter> data, seq_lengths;
};

TEST_F(ReverseSequenceV0StaticShapeInferenceTest, default_batch_seq_axes) {
    auto op = make_op(data, seq_lengths);

    input_shapes = StaticShapeVector{{4, 3, 2}, {4}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes[0], StaticShape({4, 3, 2}));
}

TEST_F(ReverseSequenceV0StaticShapeInferenceTest, set_batch_seq_axes) {
    auto op = make_op(data, seq_lengths, -1, 1);

    input_shapes = StaticShapeVector{{4, 3, 2}, {2}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes[0], StaticShape({4, 3, 2}));
}

TEST_F(ReverseSequenceV0StaticShapeInferenceTest, invalid_input_shapes_count) {
    auto op = make_op(data, seq_lengths);

    input_shapes = StaticShapeVector{{1, 2, 4}};
    EXPECT_THROW(shape_inference(op.get(), input_shapes), NodeValidationFailure);
}

TEST_F(ReverseSequenceV0StaticShapeInferenceTest, invalid_data_shape_rank) {
    auto op = make_op(data, seq_lengths);

    input_shapes = StaticShapeVector{{4}, {4}};
    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes),
                    NodeValidationFailure,
                    HasSubstr("Data input rank should be equal or greater than 2. Got: "));
}

TEST_F(ReverseSequenceV0StaticShapeInferenceTest, invalid_sequence_shape_rank) {
    auto op = make_op(data, seq_lengths);

    input_shapes = StaticShapeVector{{4, 5, 6}, {2, 2}};
    OV_EXPECT_THROW(shape_inference(op.get(), input_shapes),
                    NodeValidationFailure,
                    HasSubstr("Sequence lengths rank must be equal to 1. Got: "));
}

TEST_F(ReverseSequenceV0StaticShapeInferenceTest, default_ctor) {
    auto op = make_op();

    input_shapes = StaticShapeVector{{11, 2, 3}, {11}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes[0], StaticShape({11, 2, 3}));
}
