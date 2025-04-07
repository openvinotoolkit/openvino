// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gmock/gmock.h>

#include <array>

#include "common_test_utils/test_assertions.hpp"
#include "embeddingbag_packed_shape_inference.hpp"
#include "openvino/opsets/opset10.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace ov::opset10;
using namespace testing;

class EmbeddingBagPackedSumV3StaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v3::EmbeddingBagPackedSum> {
protected:
    void SetUp() override {
        output_shapes.resize(1);
    }
};

TEST_F(EmbeddingBagPackedSumV3StaticShapeInferenceTest, default_ctor) {
    const auto op = make_op();

    const auto batch = 8;
    auto expected_output = StaticShape{batch, 4, 5, 6};

    // 2 inputs
    {
        input_shapes = {StaticShape{3, 4, 5, 6}, StaticShape{batch, 2}};
        output_shapes = shape_inference(op.get(), input_shapes);
        EXPECT_EQ(output_shapes[0], expected_output);
    }
    // 3 inputs
    {
        input_shapes = {StaticShape{3, 4, 5, 6}, StaticShape{batch, 2}, StaticShape{batch, 2}};
        output_shapes = shape_inference(op.get(), input_shapes);
        EXPECT_EQ(output_shapes[0], expected_output);
    }
}


TEST_F(EmbeddingBagPackedSumV3StaticShapeInferenceTest, basic_2in) {
    auto emb_table = std::make_shared<Parameter>(element::f32, ov::PartialShape::dynamic());
    auto indices = std::make_shared<Parameter>(element::i64, ov::PartialShape::dynamic());

    auto op = make_op(emb_table, indices);

    input_shapes = {StaticShape{5, 2, 6}, StaticShape{3, 4}};
    output_shapes = shape_inference(op.get(), input_shapes);
    EXPECT_EQ(output_shapes[0], (StaticShape{3, 2, 6}));
}

TEST_F(EmbeddingBagPackedSumV3StaticShapeInferenceTest, basic_3in) {
    auto emb_table = std::make_shared<Parameter>(element::f32, ov::PartialShape::dynamic());
    auto indices = std::make_shared<Parameter>(element::i64, ov::PartialShape::dynamic());
    auto per_sample_weights = std::make_shared<Parameter>(element::f32, ov::PartialShape::dynamic());

    auto op = make_op(emb_table, indices, per_sample_weights);

    input_shapes = {StaticShape{5, 2, 6}, StaticShape{3, 4}, StaticShape{3, 4}};
    output_shapes = shape_inference(op.get(), input_shapes);
    EXPECT_EQ(output_shapes[0], (StaticShape{3, 2, 6}));
}
