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

class DepthToSpaceV0StaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v0::DepthToSpace> {
protected:
    void SetUp() override {
        input_shapes = {StaticShape{1, 16, 3, 1080, 1616}};
        output_shapes.resize(1);
    }
};

TEST_F(DepthToSpaceV0StaticShapeInferenceTest, default_ctor) {
    const auto op = make_op();
    op->set_block_size(2);

    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], (StaticShape{1, 2, 2 * 3, 2 * 1080, 2 * 1616}));
}

TEST_F(DepthToSpaceV0StaticShapeInferenceTest, block_first) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(4));
    const auto op = make_op(data, op_type::DepthToSpaceMode::BLOCKS_FIRST, 2);

    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], (StaticShape{1, 2, 2 * 3, 2 * 1080, 2 * 1616}));
}
