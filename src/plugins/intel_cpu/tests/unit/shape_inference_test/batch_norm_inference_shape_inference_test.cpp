// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/batch_norm.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

class BatchNormInferenceV0StaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v0::BatchNormInference> {
protected:
    void SetUp() override {
        output_shapes.resize(1);
    }
};

TEST_F(BatchNormInferenceV0StaticShapeInferenceTest, default_ctor) {
    const auto op = make_op();

    input_shapes = {{3, 5, 7}, {5, 7}, {5, 7}, {5, 7}};
    output_shapes = shape_inference(op.get(), input_shapes);
    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], (StaticShape{3, 5, 7}));
}
