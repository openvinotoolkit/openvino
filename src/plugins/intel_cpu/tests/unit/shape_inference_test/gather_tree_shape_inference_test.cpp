// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "gather_tree_shape_inference.hpp"

#include <gtest/gtest.h>

#include "openvino/op/ops.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

class GatherTreeStaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v1::GatherTree> {};

TEST_F(GatherTreeStaticShapeInferenceTest, gather_tree) {
    auto step_ids = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1});
    auto parent_idx = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1});
    auto max_seq_len = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1});
    auto end_token = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{});
    op = make_op(step_ids, parent_idx, max_seq_len, end_token);

    input_shapes = {StaticShape{1, 2, 3}, StaticShape{1, 2, 3}, StaticShape{2}, StaticShape{}};
    output_shapes = {StaticShape{}};
    output_shapes = shape_inference(op.get(), input_shapes);
    EXPECT_EQ(output_shapes[0], (StaticShape{1, 2, 3}));
}

TEST_F(GatherTreeStaticShapeInferenceTest, gather_tree_default_ctor) {
    op = make_op();
    input_shapes = {StaticShape{2, 4, 3}, StaticShape{2, 4, 3}, StaticShape{4}, StaticShape{}};
    output_shapes = {StaticShape{}};
    output_shapes = shape_inference(op.get(), input_shapes);
    EXPECT_EQ(output_shapes[0], (StaticShape{2, 4, 3}));
}
