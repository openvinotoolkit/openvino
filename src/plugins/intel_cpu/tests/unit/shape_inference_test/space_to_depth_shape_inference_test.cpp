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

class SpaceToDepthV0StaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v0::SpaceToDepth> {
protected:
    void SetUp() override {
        output_shapes.resize(1);
    }
};

TEST_F(SpaceToDepthV0StaticShapeInferenceTest, default_ctor) {
    const auto op = make_op();
    op->set_block_size(2);

    input_shapes = {StaticShape{1, 12, 4, 1080, 1616}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], (StaticShape{1, 12 * 8, 4 / 2, 1080 / 2, 1616 / 2}));
}

TEST_F(SpaceToDepthV0StaticShapeInferenceTest, depth_first_block_2) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(4));
    const auto op = make_op(data, op_type::SpaceToDepthMode::DEPTH_FIRST, 2);

    input_shapes = {StaticShape{1, 12, 4, 1080, 1616}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], (StaticShape{1, 12 * 8, 4 / 2, 1080 / 2, 1616 / 2}));
}
