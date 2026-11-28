// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "intel_gpu/op/moe_scatter_reduce.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/parameter.hpp"

using ov::intel_gpu::op::MoEScatterReduce;

namespace ov::test::intel_gpu {

namespace {

std::shared_ptr<MoEScatterReduce> make_combine(const ov::PartialShape& data_shape,
                                               const ov::PartialShape& weights_shape,
                                               const ov::PartialShape& lut_shape,
                                               const ov::element::Type& data_type = ov::element::f16,
                                               const ov::element::Type& weights_type = ov::element::f16,
                                               const ov::element::Type& lut_type = ov::element::i32) {
    return std::make_shared<MoEScatterReduce>(std::make_shared<ov::op::v0::Parameter>(data_type, data_shape),
                                              std::make_shared<ov::op::v0::Parameter>(weights_type, weights_shape),
                                              std::make_shared<ov::op::v0::Parameter>(lut_type, lut_shape));
}

}  // namespace

TEST(MoEScatterReduceOp, infers_output_type_and_shape) {
    auto combine = make_combine({15, 64}, {5, 3}, {15});
    EXPECT_EQ(combine->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(combine->get_output_partial_shape(0), (ov::PartialShape{5, 64}));
}

TEST(MoEScatterReduceOp, dynamic_tokens_give_dynamic_tokens) {
    auto combine = make_combine({-1, 64}, {-1, 6}, {-1}, ov::element::f32, ov::element::f32);
    EXPECT_EQ(combine->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(combine->get_output_partial_shape(0), (ov::PartialShape{-1, 64}));
}

TEST(MoEScatterReduceOp, dynamic_ranks_give_dynamic_dims) {
    auto combine = make_combine(ov::PartialShape::dynamic(), ov::PartialShape::dynamic(), ov::PartialShape::dynamic());
    EXPECT_EQ(combine->get_output_partial_shape(0), (ov::PartialShape{-1, -1}));
}

TEST(MoEScatterReduceOp, accepts_i32_and_i64_row_lut) {
    EXPECT_NO_THROW(make_combine({-1, 64}, {-1, 2}, {-1}, ov::element::f16, ov::element::f16, ov::element::i32));
    EXPECT_NO_THROW(make_combine({-1, 64}, {-1, 2}, {-1}, ov::element::f16, ov::element::f16, ov::element::i64));
}

TEST(MoEScatterReduceOp, rejects_other_row_lut_types) {
    EXPECT_THROW(make_combine({-1, 64}, {-1, 2}, {-1}, ov::element::f16, ov::element::f16, ov::element::u8), ov::NodeValidationFailure);
    EXPECT_THROW(make_combine({-1, 64}, {-1, 2}, {-1}, ov::element::f16, ov::element::f16, ov::element::f32), ov::NodeValidationFailure);
}

TEST(MoEScatterReduceOp, rejects_different_data_and_weights_types) {
    EXPECT_THROW(make_combine({-1, 64}, {-1, 2}, {-1}, ov::element::f16, ov::element::f32), ov::NodeValidationFailure);
}

TEST(MoEScatterReduceOp, rejects_wrong_ranks) {
    EXPECT_THROW(make_combine({-1, 1, 64}, {-1, 2}, {-1}), ov::NodeValidationFailure);
    EXPECT_THROW(make_combine({-1, 64}, {-1}, {-1}), ov::NodeValidationFailure);
    EXPECT_THROW(make_combine({-1, 64}, {-1, 2}, {-1, 2}), ov::NodeValidationFailure);
}

TEST(MoEScatterReduceOp, clone_keeps_output) {
    auto combine = make_combine({-1, 64}, {-1, 2}, {-1});
    auto clone = ov::as_type_ptr<MoEScatterReduce>(combine->clone_with_new_inputs(combine->input_values()));
    ASSERT_NE(clone, nullptr);
    EXPECT_EQ(clone->get_output_partial_shape(0), combine->get_output_partial_shape(0));
}

}  // namespace ov::test::intel_gpu
