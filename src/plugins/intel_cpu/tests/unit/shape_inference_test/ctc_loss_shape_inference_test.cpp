// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "ctc_loss_shape_inference.hpp"
#include "openvino/opsets/opset10.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::opset10;
using namespace ov::intel_cpu;

class CTCLossV4StaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v4::CTCLoss> {
protected:
    void SetUp() override {
        output_shapes.resize(1);
    }
};

TEST_F(CTCLossV4StaticShapeInferenceTest, correct_input_shapes) {
    const auto& logits = std::make_shared<Parameter>(element::f32, PartialShape{-1, -1, -1});
    const auto& logit_length = std::make_shared<Parameter>(element::i32, PartialShape{-1});
    const auto& labels = std::make_shared<Parameter>(element::i32, PartialShape{-1, -1});
    const auto& label_length = std::make_shared<Parameter>(element::i32, PartialShape{-1});
    const auto& blank_index = std::make_shared<Parameter>(element::i32, ov::Shape{});

    auto op = make_op(logits, logit_length, labels, label_length, blank_index);

    input_shapes = StaticShapeVector{{10, 120, 28}, {10}, {10, 120}, {10}, {}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({10}));
}

TEST_F(CTCLossV4StaticShapeInferenceTest, default_ctor) {
    auto op = make_op();

    input_shapes = StaticShapeVector{{12, 120, 28}, {12}, {12, 120}, {12}, {}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({12}));
}
