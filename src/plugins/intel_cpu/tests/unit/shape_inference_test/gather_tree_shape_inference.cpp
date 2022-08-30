// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <gather_tree_shape_inference.hpp>
#include <openvino/op/gather_tree.hpp>
#include <openvino/op/ops.hpp>
#include <openvino/op/parameter.hpp>
#include <utils/shape_inference/shape_inference.hpp>
#include <utils/shape_inference/static_shape.hpp>

using namespace ov;
using namespace ov::intel_cpu;

TEST(StaticShapeInferenceTest, GatherTreeTest) {
    auto step_ids = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1});
    auto parent_idx = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1});
    auto max_seq_len = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1});
    auto end_token = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{Shape{}});
    auto gather_tree = std::make_shared<op::v1::GatherTree>(step_ids, parent_idx, max_seq_len, end_token);
    // Test StaticShape
    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 2, 3},
                                                    StaticShape{1, 2, 3},
                                                    StaticShape{2},
                                                    StaticShape{}},
                             static_output_shapes = {StaticShape{}};
    shape_inference(gather_tree.get(), static_input_shapes, static_output_shapes);
    ASSERT_EQ(static_output_shapes[0], (StaticShape{1, 2, 3}));
}