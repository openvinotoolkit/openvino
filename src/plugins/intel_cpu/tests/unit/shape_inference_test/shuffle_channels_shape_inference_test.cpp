// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/opsets/opset10.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace ov::opset10;
using namespace testing;

class ShuffleChannelsV0StaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v0::ShuffleChannels> {
protected:
    void SetUp() override {
        output_shapes.resize(1);
    }
};

TEST_F(ShuffleChannelsV0StaticShapeInferenceTest, default_ctor) {
    op = make_op();
    op->set_axis(-2);
    op->set_group(2);

    input_shapes = {StaticShape{5, 4, 9}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], input_shapes[0]);
}

TEST_F(ShuffleChannelsV0StaticShapeInferenceTest, correct_shape_infer) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{-1, -1, -1});
    op = make_op(data, -1, 3);

    input_shapes = {StaticShape{5, 4, 9}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes[0], input_shapes[0]);
}
