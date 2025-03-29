// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "batch_norm_shape_inference.hpp"
#include "openvino/op/batch_norm.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

class BatchNormInferenceV0StaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v0::BatchNormInference> {};
class BatchNormInferenceV5StaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v5::BatchNormInference> {};

TEST_F(BatchNormInferenceV0StaticShapeInferenceTest, default_ctor_direct_infer_call) {
    const auto op = make_op();

    input_shapes = {{5}, {5}, {3, 5, 7}, {5}, {5}};
    output_shapes = op::v0::shape_infer(op.get(), input_shapes);
    ASSERT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], (StaticShape{3, 5, 7}));
}

TEST_F(BatchNormInferenceV5StaticShapeInferenceTest, default_ctor_direct_infer_call) {
    const auto op = make_op();

    input_shapes = {{3, 5, 7}, {5}, {5}, {5}, {5}};
    output_shapes = op::v5::shape_infer(op.get(), input_shapes);
    ASSERT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], (StaticShape{3, 5, 7}));
}
