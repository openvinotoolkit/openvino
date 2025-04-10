// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gmock/gmock.h>

#include <array>

#include "common_test_utils/test_assertions.hpp"
#include "embeddingbag_offsets_shape_inference.hpp"
#include "openvino/opsets/opset10.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace ov::opset10;
using namespace testing;

class EmbeddingBagOffsetsSumV3StaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v3::EmbeddingBagOffsetsSum> {
protected:
    void SetUp() override {
        output_shapes.resize(1);
    }
};

TEST_F(EmbeddingBagOffsetsSumV3StaticShapeInferenceTest, default_ctor) {
    const auto op = make_op();

    const auto batch = 8;
    auto expected_output = StaticShape{batch, 4, 5, 6};

    // 3 inputs
    {
        input_shapes = {StaticShape{3, 4, 5, 6}, StaticShape{2}, StaticShape{batch}};
        output_shapes = shape_inference(op.get(), input_shapes);
        EXPECT_EQ(output_shapes[0], expected_output);
    }
    // 4 inputs
    {
        input_shapes = {StaticShape{3, 4, 5, 6}, StaticShape{2}, StaticShape{batch}, StaticShape{}};
        output_shapes = shape_inference(op.get(), input_shapes);
        EXPECT_EQ(output_shapes[0], expected_output);
    }
    // 5 inputs
    {
        input_shapes = {StaticShape{3, 4, 5, 6}, StaticShape{2}, StaticShape{batch}, StaticShape{}, StaticShape{2}};
        output_shapes = shape_inference(op.get(), input_shapes);
        EXPECT_EQ(output_shapes[0], expected_output);
    }
}

TEST_F(EmbeddingBagOffsetsSumV3StaticShapeInferenceTest, basic_3in) {
    auto emb_table = std::make_shared<Parameter>(element::f32, ov::PartialShape::dynamic());
    auto indices = std::make_shared<Parameter>(element::i64, ov::PartialShape::dynamic());
    auto offsets = std::make_shared<Parameter>(element::i64, ov::PartialShape::dynamic());

    auto op = make_op(emb_table, indices, offsets);

    auto expected_output = StaticShape{3, 2, 6};

    input_shapes = {StaticShape{5, 2, 6}, StaticShape{4}, StaticShape{3}};
    output_shapes = shape_inference(op.get(), input_shapes);
    EXPECT_EQ(output_shapes[0], expected_output);
}

TEST_F(EmbeddingBagOffsetsSumV3StaticShapeInferenceTest, basic_4in) {
    auto emb_table = std::make_shared<Parameter>(element::f32, ov::PartialShape::dynamic());
    auto indices = std::make_shared<Parameter>(element::i64, ov::PartialShape::dynamic());
    auto offsets = std::make_shared<Parameter>(element::i64, ov::PartialShape::dynamic());
    auto default_index = std::make_shared<Parameter>(element::i64, ov::PartialShape::dynamic());

    auto op = make_op(emb_table, indices, offsets, default_index);

    auto expected_output = StaticShape{3, 2, 6};

    input_shapes = {StaticShape{5, 2, 6}, StaticShape{4}, StaticShape{3}, StaticShape{}};
    output_shapes = shape_inference(op.get(), input_shapes);
    EXPECT_EQ(output_shapes[0], expected_output);
}

TEST_F(EmbeddingBagOffsetsSumV3StaticShapeInferenceTest, basic_5in) {
    auto emb_table = std::make_shared<Parameter>(element::f32, ov::PartialShape::dynamic());
    auto indices = std::make_shared<Parameter>(element::i64, ov::PartialShape::dynamic());
    auto offsets = std::make_shared<Parameter>(element::i64, ov::PartialShape::dynamic());
    auto default_index = std::make_shared<Parameter>(element::i64, ov::PartialShape::dynamic());
    auto per_sample_weights = std::make_shared<Parameter>(element::f32, ov::PartialShape::dynamic());

    auto op = make_op(emb_table, indices, offsets, default_index, per_sample_weights);

    auto expected_output = StaticShape{3, 2, 6};

    input_shapes = {StaticShape{5, 2, 6}, StaticShape{4}, StaticShape{3}, StaticShape{}, StaticShape{4}};
    output_shapes = shape_inference(op.get(), input_shapes);
    EXPECT_EQ(output_shapes[0], expected_output);
}
