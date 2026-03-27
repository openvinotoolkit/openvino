// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/gather_matmul.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "ov_ops/gather_matmul_compressed.hpp"

using namespace ov;
using namespace ov::op;
using GatherMatmul = ov::op::internal::GatherMatmul;
using GatherMatmulCompressed = ov::op::internal::GatherMatmulCompressed;

namespace {

auto make_const(element::Type et, const Shape& shape) {
    return v0::Constant::create(et, shape, std::vector<float>(shape_size(shape), 0.f));
}

auto make_param(element::Type et, const PartialShape& shape) {
    return std::make_shared<v0::Parameter>(et, shape);
}

// No-bias placeholder (matches 3-arg constructor's internal representation)
auto make_empty_bias() {
    return std::make_shared<v0::Constant>(element::dynamic, Shape{0});
}

}  // namespace

// ============================================================================
// Positive tests — GatherMatmul shape inference
// ============================================================================

// 3-arg constructor: A[0]=1 broadcasts to topk=2
// Gate projection: [1,64,2048] x [8,4096,2048]^T → [2,64,4096]
TEST(GatherMatmulTest, shape_3arg_gate_broadcast) {
    auto A = make_param(element::f32, {1, 64, 2048});
    auto B = make_const(element::f32, {8, 4096, 2048});
    auto idx = make_param(element::i32, {64, 2});

    auto op = std::make_shared<GatherMatmul>(A, B, idx);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 64, 4096}));
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
}

// 3-arg constructor: A[0]=topk (non-broadcast)
// Down projection: [2,64,4096] x [8,2048,4096]^T → [2,64,2048]
TEST(GatherMatmulTest, shape_3arg_down_projection) {
    auto A = make_param(element::f32, {2, 64, 4096});
    auto B = make_const(element::f32, {8, 2048, 4096});
    auto idx = make_param(element::i32, {64, 2});

    auto op = std::make_shared<GatherMatmul>(A, B, idx);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 64, 2048}));
}

// 4-arg constructor with 3D bias [n_experts, 1, N]
TEST(GatherMatmulTest, shape_4arg_bias_3d) {
    auto A = make_param(element::f32, {1, 64, 2048});
    auto B = make_const(element::f32, {8, 8192, 2048});
    auto idx = make_param(element::i32, {64, 2});
    auto bias = make_const(element::f32, {8, 1, 8192});

    auto op = std::make_shared<GatherMatmul>(A, B, idx, bias);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 64, 8192}));
}

// 4-arg constructor with 1D bias
TEST(GatherMatmulTest, shape_4arg_bias_1d) {
    auto A = make_param(element::f32, {1, 64, 2048});
    auto B = make_const(element::f32, {8, 4096, 2048});
    auto idx = make_param(element::i32, {64, 2});
    auto bias = make_const(element::f32, {4096});

    auto op = std::make_shared<GatherMatmul>(A, B, idx, bias);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 64, 4096}));
}

// 4-arg constructor with scalar bias
TEST(GatherMatmulTest, shape_4arg_bias_scalar) {
    auto A = make_param(element::f32, {1, 64, 2048});
    auto B = make_const(element::f32, {8, 4096, 2048});
    auto idx = make_param(element::i32, {64, 2});
    auto bias = make_const(element::f32, {});

    auto op = std::make_shared<GatherMatmul>(A, B, idx, bias);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 64, 4096}));
}

// 4D group-compressed weights: [8, 4096, 16, 128] → effective K = 16*128 = 2048
TEST(GatherMatmulTest, shape_weights_4d_grouped) {
    auto A = make_param(element::f32, {1, 64, 2048});
    auto B = make_const(element::f32, {8, 4096, 16, 128});
    auto idx = make_param(element::i32, {64, 2});

    auto op = std::make_shared<GatherMatmul>(A, B, idx);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 64, 4096}));
}

// Output element type should match input A
TEST(GatherMatmulTest, output_type_matches_A) {
    auto A = make_param(element::f16, {1, 64, 2048});
    auto B = make_const(element::f16, {8, 4096, 2048});
    auto idx = make_param(element::i32, {64, 2});

    auto op = std::make_shared<GatherMatmul>(A, B, idx);

    EXPECT_EQ(op->get_output_element_type(0), element::f16);
}

// Dynamic sequence length in A and indices
TEST(GatherMatmulTest, shape_dynamic_seq_len) {
    auto A = make_param(element::f32, {1, Dimension::dynamic(), 2048});
    auto B = make_const(element::f32, {8, 4096, 2048});
    auto idx = make_param(element::i32, {Dimension::dynamic(), 2});

    auto op = std::make_shared<GatherMatmul>(A, B, idx);

    auto out = op->get_output_partial_shape(0);
    EXPECT_EQ(out[0], 2);
    EXPECT_TRUE(out[1].is_dynamic());
    EXPECT_EQ(out[2], 4096);
}

// Dynamic topk dimension
TEST(GatherMatmulTest, shape_dynamic_topk) {
    auto A = make_param(element::f32, {Dimension::dynamic(), 64, 2048});
    auto B = make_const(element::f32, {8, 4096, 2048});
    auto idx = make_param(element::i32, {64, Dimension::dynamic()});

    auto op = std::make_shared<GatherMatmul>(A, B, idx);

    auto out = op->get_output_partial_shape(0);
    EXPECT_TRUE(out[0].is_dynamic());
    EXPECT_EQ(out[1], 64);
    EXPECT_EQ(out[2], 4096);
}

// Edge case: single expert, topk=1
TEST(GatherMatmulTest, shape_single_expert) {
    auto A = make_param(element::f32, {1, 32, 1024});
    auto B = make_const(element::f32, {1, 512, 1024});
    auto idx = make_param(element::i32, {32, 1});

    auto op = std::make_shared<GatherMatmul>(A, B, idx);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1, 32, 512}));
}

// ============================================================================
// Positive tests — GatherMatmulCompressed
// ============================================================================

// Basic 3D compressed weights
TEST(GatherMatmulCompressedTest, shape_basic_3d) {
    auto A = make_param(element::f32, {1, 64, 2048});
    auto B = make_const(element::u8, {8, 4096, 2048});
    auto idx = make_param(element::i32, {64, 2});
    auto bias = make_empty_bias();
    auto scales = make_const(element::f32, {8, 4096, 1});
    auto zp = make_const(element::u8, {8, 4096, 1});

    auto op = std::make_shared<GatherMatmulCompressed>(A, B, idx, bias, scales, zp);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 64, 4096}));
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
}

// 4D group-compressed weights with scales/zp
TEST(GatherMatmulCompressedTest, shape_grouped_4d) {
    auto A = make_param(element::f32, {1, 64, 2048});
    auto B = make_const(element::u8, {8, 4096, 16, 128});
    auto idx = make_param(element::i32, {64, 2});
    auto bias = make_empty_bias();
    auto scales = make_const(element::f32, {8, 4096, 16, 1});
    auto zp = make_const(element::u8, {8, 4096, 16, 1});

    auto op = std::make_shared<GatherMatmulCompressed>(A, B, idx, bias, scales, zp);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 64, 4096}));
}

// ============================================================================
// Negative tests — GatherMatmul validation failures
// ============================================================================

TEST(GatherMatmulTest, fail_B_not_constant) {
    auto A = make_param(element::f32, {1, 64, 2048});
    auto B = make_param(element::f32, {8, 4096, 2048});  // Parameter, not Constant
    auto idx = make_param(element::i32, {64, 2});

    OV_EXPECT_THROW((void)std::make_shared<GatherMatmul>(A, B, idx),
                    ov::NodeValidationFailure,
                    testing::HasSubstr("Input B must be on constant path"));
}

TEST(GatherMatmulTest, fail_A_rank_2d) {
    auto A = make_param(element::f32, {64, 2048});
    auto B = make_const(element::f32, {8, 4096, 2048});
    auto idx = make_param(element::i32, {64, 2});

    OV_EXPECT_THROW((void)std::make_shared<GatherMatmul>(A, B, idx),
                    ov::NodeValidationFailure,
                    testing::HasSubstr("Input A rank must be exactly 3D"));
}

TEST(GatherMatmulTest, fail_A_rank_4d) {
    auto A = make_param(element::f32, {1, 1, 64, 2048});
    auto B = make_const(element::f32, {8, 4096, 2048});
    auto idx = make_param(element::i32, {64, 2});

    OV_EXPECT_THROW((void)std::make_shared<GatherMatmul>(A, B, idx),
                    ov::NodeValidationFailure,
                    testing::HasSubstr("Input A rank must be exactly 3D"));
}

TEST(GatherMatmulTest, fail_B_rank_2d) {
    auto A = make_param(element::f32, {1, 64, 2048});
    auto B = make_const(element::f32, {4096, 2048});
    auto idx = make_param(element::i32, {64, 2});

    OV_EXPECT_THROW((void)std::make_shared<GatherMatmul>(A, B, idx),
                    ov::NodeValidationFailure,
                    testing::HasSubstr("Input B rank must be 3D or 4D"));
}

TEST(GatherMatmulTest, fail_B_rank_5d) {
    auto A = make_param(element::f32, {1, 64, 2048});
    auto B = make_const(element::f32, {8, 4096, 4, 4, 32});
    auto idx = make_param(element::i32, {64, 2});

    OV_EXPECT_THROW((void)std::make_shared<GatherMatmul>(A, B, idx),
                    ov::NodeValidationFailure,
                    testing::HasSubstr("Input B rank must be 3D or 4D"));
}

TEST(GatherMatmulTest, fail_indices_rank_1d) {
    auto A = make_param(element::f32, {1, 64, 2048});
    auto B = make_const(element::f32, {8, 4096, 2048});
    auto idx = make_param(element::i32, {64});

    OV_EXPECT_THROW((void)std::make_shared<GatherMatmul>(A, B, idx),
                    ov::NodeValidationFailure,
                    testing::HasSubstr("Input indices rank must be exactly 2D"));
}

TEST(GatherMatmulTest, fail_indices_rank_3d) {
    auto A = make_param(element::f32, {1, 64, 2048});
    auto B = make_const(element::f32, {8, 4096, 2048});
    auto idx = make_param(element::i32, {1, 64, 2});

    OV_EXPECT_THROW((void)std::make_shared<GatherMatmul>(A, B, idx),
                    ov::NodeValidationFailure,
                    testing::HasSubstr("Input indices rank must be exactly 2D"));
}

TEST(GatherMatmulTest, fail_bias_rank_2d) {
    auto A = make_param(element::f32, {1, 64, 2048});
    auto B = make_const(element::f32, {8, 4096, 2048});
    auto idx = make_param(element::i32, {64, 2});
    auto bias = make_const(element::f32, {1, 4096});

    OV_EXPECT_THROW((void)std::make_shared<GatherMatmul>(A, B, idx, bias),
                    ov::NodeValidationFailure,
                    testing::HasSubstr("Input bias rank must be either 1D, scalar"));
}

// A[0]=3 but topk=2 → 3 is neither 1 nor 2
TEST(GatherMatmulTest, fail_A0_not_broadcastable_to_topk) {
    auto A = make_param(element::f32, {3, 64, 2048});
    auto B = make_const(element::f32, {8, 4096, 2048});
    auto idx = make_param(element::i32, {64, 2});

    OV_EXPECT_THROW((void)std::make_shared<GatherMatmul>(A, B, idx),
                    ov::NodeValidationFailure,
                    testing::HasSubstr("first dimension of input A must be equal"));
}

// topk=8 > n_experts=4
TEST(GatherMatmulTest, fail_topk_exceeds_experts) {
    auto A = make_param(element::f32, {1, 64, 2048});
    auto B = make_const(element::f32, {4, 4096, 2048});
    auto idx = make_param(element::i32, {64, 8});

    OV_EXPECT_THROW((void)std::make_shared<GatherMatmul>(A, B, idx),
                    ov::NodeValidationFailure,
                    testing::HasSubstr("second dimension of input indices must be less"));
}

// A[1]=64 but idx[0]=32
TEST(GatherMatmulTest, fail_seq_length_mismatch) {
    auto A = make_param(element::f32, {1, 64, 2048});
    auto B = make_const(element::f32, {8, 4096, 2048});
    auto idx = make_param(element::i32, {32, 2});

    OV_EXPECT_THROW((void)std::make_shared<GatherMatmul>(A, B, idx),
                    ov::NodeValidationFailure,
                    testing::HasSubstr("first dimension of input indices must be equal"));
}

// ============================================================================
// Negative tests — GatherMatmulCompressed validation failures
// ============================================================================

TEST(GatherMatmulCompressedTest, fail_scales_not_constant) {
    auto A = make_param(element::f32, {1, 64, 2048});
    auto B = make_const(element::u8, {8, 4096, 2048});
    auto idx = make_param(element::i32, {64, 2});
    auto bias = make_empty_bias();
    auto scales = make_param(element::f32, {8, 4096, 1});  // Parameter, not Constant
    auto zp = make_const(element::u8, {8, 4096, 1});

    OV_EXPECT_THROW((void)std::make_shared<GatherMatmulCompressed>(A, B, idx, bias, scales, zp),
                    ov::NodeValidationFailure,
                    testing::HasSubstr("Input weight_scales must be a Constant"));
}

TEST(GatherMatmulCompressedTest, fail_zp_not_constant) {
    auto A = make_param(element::f32, {1, 64, 2048});
    auto B = make_const(element::u8, {8, 4096, 2048});
    auto idx = make_param(element::i32, {64, 2});
    auto bias = make_empty_bias();
    auto scales = make_const(element::f32, {8, 4096, 1});
    auto zp = make_param(element::u8, {8, 4096, 1});  // Parameter, not Constant

    OV_EXPECT_THROW((void)std::make_shared<GatherMatmulCompressed>(A, B, idx, bias, scales, zp),
                    ov::NodeValidationFailure,
                    testing::HasSubstr("Input weight_zero_points must be a Constant"));
}

// ============================================================================
// Clone tests
// ============================================================================

TEST(GatherMatmulTest, clone_preserves_output_shape) {
    auto A = make_param(element::f32, {1, 64, 2048});
    auto B = make_const(element::f32, {8, 4096, 2048});
    auto idx = make_param(element::i32, {64, 2});

    auto op = std::make_shared<GatherMatmul>(A, B, idx);
    auto cloned = op->clone_with_new_inputs({A, B, idx, make_empty_bias()});

    EXPECT_EQ(cloned->get_output_partial_shape(0), op->get_output_partial_shape(0));
    EXPECT_EQ(cloned->get_output_element_type(0), op->get_output_element_type(0));
}

TEST(GatherMatmulCompressedTest, clone_preserves_output_shape) {
    auto A = make_param(element::f32, {1, 64, 2048});
    auto B = make_const(element::u8, {8, 4096, 2048});
    auto idx = make_param(element::i32, {64, 2});
    auto bias = make_empty_bias();
    auto scales = make_const(element::f32, {8, 4096, 1});
    auto zp = make_const(element::u8, {8, 4096, 1});

    auto op = std::make_shared<GatherMatmulCompressed>(A, B, idx, bias, scales, zp);
    auto cloned = op->clone_with_new_inputs({A, B, idx, bias, scales, zp});

    EXPECT_EQ(cloned->get_output_partial_shape(0), op->get_output_partial_shape(0));
    EXPECT_EQ(cloned->get_output_element_type(0), op->get_output_element_type(0));
}
