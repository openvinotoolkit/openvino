// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ctc_loss_shape_inference.hpp>
#include <openvino/op/ctc_loss.hpp>
#include <openvino/op/parameter.hpp>
#include <utils/shape_inference/shape_inference.hpp>
#include <utils/shape_inference/static_shape.hpp>

using namespace ov;
using namespace ov::intel_cpu;

TEST(StaticShapeInferenceTest, CTCLossTest) {
    const auto& logits = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1});
    const auto& logit_length = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto& labels = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1, -1});
    const auto& label_length = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto& blank_index = std::make_shared<ov::op::v0::Parameter>(element::i32, ov::Shape{});

    // create CTCLoss node
    auto ctc_loss = std::make_shared<op::v4::CTCLoss>(logits, logit_length, labels, label_length, blank_index);

    std::vector<StaticShape> static_input_shapes = {StaticShape{10, 120, 28},
                                                    StaticShape{10},
                                                    StaticShape{10, 120},
                                                    StaticShape{10},
                                                    ov::Shape{}},
                             static_output_shapes = {StaticShape{}};
    shape_inference(ctc_loss.get(), static_input_shapes, static_output_shapes);

    ASSERT_EQ(static_output_shapes[0], StaticShape({10}));
}