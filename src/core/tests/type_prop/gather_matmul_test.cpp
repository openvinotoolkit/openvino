// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/gather_matmul.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"

using ov::op::v0::Constant;
using ov::op::v0::Parameter;

namespace ov::tests {

namespace {
auto make_const(element::Type et, const Shape& shape) {
    return Constant::create(et, shape, std::vector<float>(shape_size(shape), 0.f));
}

auto make_param(element::Type et, const PartialShape& shape) {
    return std::make_shared<Parameter>(et, shape);
}

auto make_empty_bias() {
    return std::make_shared<Constant>(element::dynamic, Shape{0});
}
}  // namespace

using GatherMatmulTest = TypePropOpTest<ov::op::internal::GatherMatmul>;

// ============================================================================
// Positive tests — GatherMatmul shape inference
// ============================================================================

// 3-arg constructor: A[0]=1 broadcasts to topk=2
TEST_F(GatherMatmulTest, shape_3arg_gate_broadcast) {
    auto a = make_param(element::f32, {1, 64, 2048});
    auto b = make_const(element::f32, {8, 4096, 2048});
    auto idx = make_param(element::i32, {64, 2});

    auto op = make_op(a, b, idx);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 64, 4096}));
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
}

// 3-arg constructor: A[0]=topk (non-broadcast)
TEST_F(GatherMatmulTest, shape_3arg_down_projection) {
    auto a = make_param(element::f32, {2, 64, 4096});
    auto b = make_const(element::f32, {8, 2048, 4096});
    auto idx = make_param(element::i32, {64, 2});

    auto op = make_op(a, b, idx);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 64, 2048}));
}

// 4-arg constructor with 3D bias [n_experts, 1, N]
TEST_F(GatherMatmulTest, shape_4arg_bias_3d) {
    auto a = make_param(element::f32, {1, 64, 2048});
    auto b = make_const(element::f32, {8, 8192, 2048});
    auto idx = make_param(element::i32, {64, 2});
    auto bias = make_const(element::f32, {8, 1, 8192});

    auto op = make_op(a, b, idx, bias);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 64, 8192}));
}

// 4-arg constructor with 1D bias
TEST_F(GatherMatmulTest, shape_4arg_bias_1d) {
    auto a = make_param(element::f32, {1, 64, 2048});
    auto b = make_const(element::f32, {8, 4096, 2048});
    auto idx = make_param(element::i32, {64, 2});
    auto bias = make_const(element::f32, {4096});

    auto op = make_op(a, b, idx, bias);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 64, 4096}));
}

// 4-arg constructor with scalar bias
TEST_F(GatherMatmulTest, shape_4arg_bias_scalar) {
    auto a = make_param(element::f32, {1, 64, 2048});
    auto b = make_const(element::f32, {8, 4096, 2048});
    auto idx = make_param(element::i32, {64, 2});
    auto bias = make_const(element::f32, {});

    auto op = make_op(a, b, idx, bias);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 64, 4096}));
}

// 4D group-compressed weights
TEST_F(GatherMatmulTest, shape_weights_4d_grouped) {
    auto a = make_param(element::f32, {1, 64, 2048});
    auto b = make_const(element::f32, {8, 4096, 16, 128});
    auto idx = make_param(element::i32, {64, 2});

    auto op = make_op(a, b, idx);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 64, 4096}));
}

// Output element type should match input A
TEST_F(GatherMatmulTest, output_type_matches_A) {
    auto a = make_param(element::f16, {1, 64, 2048});
    auto b = make_const(element::f16, {8, 4096, 2048});
    auto idx = make_param(element::i32, {64, 2});

    auto op = make_op(a, b, idx);

    EXPECT_EQ(op->get_output_element_type(0), element::f16);
}

// Dynamic sequence length in A and indices
TEST_F(GatherMatmulTest, shape_dynamic_seq_len) {
    auto a = make_param(element::f32, {1, Dimension::dynamic(), 2048});
    auto b = make_const(element::f32, {8, 4096, 2048});
    auto idx = make_param(element::i32, {Dimension::dynamic(), 2});

    auto op = make_op(a, b, idx);

    const auto& out = op->get_output_partial_shape(0);
    EXPECT_EQ(out[0], 2);
    EXPECT_TRUE(out[1].is_dynamic());
    EXPECT_EQ(out[2], 4096);
}

// Dynamic topk dimension
TEST_F(GatherMatmulTest, shape_dynamic_topk) {
    auto a = make_param(element::f32, {Dimension::dynamic(), 64, 2048});
    auto b = make_const(element::f32, {8, 4096, 2048});
    auto idx = make_param(element::i32, {64, Dimension::dynamic()});

    auto op = make_op(a, b, idx);

    const auto& out = op->get_output_partial_shape(0);
    EXPECT_TRUE(out[0].is_dynamic());
    EXPECT_EQ(out[1], 64);
    EXPECT_EQ(out[2], 4096);
}

// Edge case: single expert, topk=1
TEST_F(GatherMatmulTest, shape_single_expert) {
    auto a = make_param(element::f32, {1, 32, 1024});
    auto b = make_const(element::f32, {1, 512, 1024});
    auto idx = make_param(element::i32, {32, 1});

    auto op = make_op(a, b, idx);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1, 32, 512}));
}

// ============================================================================
// Negative tests — GatherMatmul validation failures
// ============================================================================

TEST_F(GatherMatmulTest, fail_A_rank_2d) {
    auto a = make_param(element::f32, {64, 2048});
    auto b = make_const(element::f32, {8, 4096, 2048});
    auto idx = make_param(element::i32, {64, 2});

    OV_EXPECT_THROW(std::ignore = make_op(a, b, idx),
                    ov::NodeValidationFailure,
                    testing::HasSubstr("Input A rank must be exactly 3D"));
}

TEST_F(GatherMatmulTest, fail_A_rank_4d) {
    auto a = make_param(element::f32, {1, 1, 64, 2048});
    auto b = make_const(element::f32, {8, 4096, 2048});
    auto idx = make_param(element::i32, {64, 2});

    OV_EXPECT_THROW(std::ignore = make_op(a, b, idx),
                    ov::NodeValidationFailure,
                    testing::HasSubstr("Input A rank must be exactly 3D"));
}

TEST_F(GatherMatmulTest, fail_B_rank_2d) {
    auto a = make_param(element::f32, {1, 64, 2048});
    auto b = make_const(element::f32, {4096, 2048});
    auto idx = make_param(element::i32, {64, 2});

    OV_EXPECT_THROW(std::ignore = make_op(a, b, idx),
                    ov::NodeValidationFailure,
                    testing::HasSubstr("Input B rank must be 3D or 4D"));
}

TEST_F(GatherMatmulTest, fail_B_rank_5d) {
    auto a = make_param(element::f32, {1, 64, 2048});
    auto b = make_const(element::f32, {8, 4096, 4, 4, 32});
    auto idx = make_param(element::i32, {64, 2});

    OV_EXPECT_THROW(std::ignore = make_op(a, b, idx),
                    ov::NodeValidationFailure,
                    testing::HasSubstr("Input B rank must be 3D or 4D"));
}

TEST_F(GatherMatmulTest, fail_indices_rank_1d) {
    auto a = make_param(element::f32, {1, 64, 2048});
    auto b = make_const(element::f32, {8, 4096, 2048});
    auto idx = make_param(element::i32, {64});

    OV_EXPECT_THROW(std::ignore = make_op(a, b, idx),
                    ov::NodeValidationFailure,
                    testing::HasSubstr("Input indices rank must be exactly 2D"));
}

TEST_F(GatherMatmulTest, fail_indices_rank_3d) {
    auto a = make_param(element::f32, {1, 64, 2048});
    auto b = make_const(element::f32, {8, 4096, 2048});
    auto idx = make_param(element::i32, {1, 64, 2});

    OV_EXPECT_THROW(std::ignore = make_op(a, b, idx),
                    ov::NodeValidationFailure,
                    testing::HasSubstr("Input indices rank must be exactly 2D"));
}

TEST_F(GatherMatmulTest, fail_bias_rank_2d) {
    auto a = make_param(element::f32, {1, 64, 2048});
    auto b = make_const(element::f32, {8, 4096, 2048});
    auto idx = make_param(element::i32, {64, 2});
    auto bias = make_const(element::f32, {1, 4096});

    OV_EXPECT_THROW(std::ignore = make_op(a, b, idx, bias),
                    ov::NodeValidationFailure,
                    testing::HasSubstr("Input bias rank must be either 1D, scalar"));
}

// A[0]=3 but topk=2 → 3 is neither 1 nor 2
TEST_F(GatherMatmulTest, fail_A0_not_broadcastable_to_topk) {
    auto a = make_param(element::f32, {3, 64, 2048});
    auto b = make_const(element::f32, {8, 4096, 2048});
    auto idx = make_param(element::i32, {64, 2});

    OV_EXPECT_THROW(std::ignore = make_op(a, b, idx),
                    ov::NodeValidationFailure,
                    testing::HasSubstr("first dimension of input A must be equal"));
}

// topk=8 > n_experts=4
TEST_F(GatherMatmulTest, fail_topk_exceeds_experts) {
    auto a = make_param(element::f32, {1, 64, 2048});
    auto b = make_const(element::f32, {4, 4096, 2048});
    auto idx = make_param(element::i32, {64, 8});

    OV_EXPECT_THROW(std::ignore = make_op(a, b, idx),
                    ov::NodeValidationFailure,
                    testing::HasSubstr("second dimension of input indices must be less"));
}

// A[1]=64 but idx[0]=32
TEST_F(GatherMatmulTest, fail_seq_length_mismatch) {
    auto a = make_param(element::f32, {1, 64, 2048});
    auto b = make_const(element::f32, {8, 4096, 2048});
    auto idx = make_param(element::i32, {32, 2});

    OV_EXPECT_THROW(std::ignore = make_op(a, b, idx),
                    ov::NodeValidationFailure,
                    testing::HasSubstr("first dimension of input indices must be equal"));
}

// ============================================================================
// Clone tests
// ============================================================================

TEST_F(GatherMatmulTest, clone_preserves_output_shape) {
    auto a = make_param(element::f32, {1, 64, 2048});
    auto b = make_const(element::f32, {8, 4096, 2048});
    auto idx = make_param(element::i32, {64, 2});

    auto op = make_op(a, b, idx);
    auto cloned = op->clone_with_new_inputs({a, b, idx, make_empty_bias()});

    EXPECT_EQ(cloned->get_output_partial_shape(0), op->get_output_partial_shape(0));
    EXPECT_EQ(cloned->get_output_element_type(0), op->get_output_element_type(0));
}

}  // namespace ov::tests
