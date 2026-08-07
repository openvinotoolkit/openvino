// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/mamba2.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/openvino.hpp"

namespace ov::test {
namespace {

// Shapes: A[H], dt[B,T,H], B[B,T,G,N], x[B,T,H,P], C[B,T,G,N], recurrent_state[B,H,P,N]
std::shared_ptr<op::internal::Mamba2> make_mamba2(const element::Type& et,
                                                  const PartialShape& A,
                                                  const PartialShape& dt,
                                                  const PartialShape& B,
                                                  const PartialShape& x,
                                                  const PartialShape& C,
                                                  const PartialShape& state) {
    auto A_p = std::make_shared<op::v0::Parameter>(et, A);
    auto dt_p = std::make_shared<op::v0::Parameter>(et, dt);
    auto B_p = std::make_shared<op::v0::Parameter>(et, B);
    auto x_p = std::make_shared<op::v0::Parameter>(et, x);
    auto C_p = std::make_shared<op::v0::Parameter>(et, C);
    auto recurrent_state = std::make_shared<op::v0::Parameter>(et, state);

    return std::make_shared<op::internal::Mamba2>(OutputVector{A_p, dt_p, B_p, x_p, C_p, recurrent_state});
}

}  // namespace

TEST(type_prop, mamba2_static_f32) {
    // B=2, T=5, H=4, G=2, P=8 (head_dim), N=16 (state_size)
    const auto op = make_mamba2(element::f32,
                                Shape{4},
                                Shape{2, 5, 4},
                                Shape{2, 5, 2, 16},
                                Shape{2, 5, 4, 8},
                                Shape{2, 5, 2, 16},
                                Shape{2, 4, 8, 16});

    EXPECT_EQ(op->get_output_size(), 2);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_element_type(1), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape(Shape{2, 5, 4, 8}));
    EXPECT_EQ(op->get_output_partial_shape(1), PartialShape(Shape{2, 4, 8, 16}));
}

TEST(type_prop, mamba2_static_f16) {
    const auto op = make_mamba2(element::f16,
                                Shape{4},
                                Shape{2, 5, 4},
                                Shape{2, 5, 2, 16},
                                Shape{2, 5, 4, 8},
                                Shape{2, 5, 2, 16},
                                Shape{2, 4, 8, 16});

    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_element_type(1), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape(Shape{2, 5, 4, 8}));
    EXPECT_EQ(op->get_output_partial_shape(1), PartialShape(Shape{2, 4, 8, 16}));
}

TEST(type_prop, mamba2_partial_shape_infer) {
    const auto op = make_mamba2(element::bf16,
                                PartialShape{4},
                                PartialShape{{1, 4}, -1, 4},
                                PartialShape{{1, 4}, -1, 2, {32, 128}},
                                PartialShape{{1, 4}, -1, 4, {2, 8}},
                                PartialShape{{1, 4}, -1, 2, {32, 128}},
                                PartialShape{{1, 4}, 4, {2, 8}, {32, 128}});

    EXPECT_EQ(op->get_output_element_type(0), element::bf16);
    EXPECT_EQ(op->get_output_element_type(1), element::bf16);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{{1, 4}, -1, 4, {2, 8}}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{{1, 4}, 4, {2, 8}, {32, 128}}));
}

TEST(type_prop, mamba2_invalid_A_rank) {
    OV_EXPECT_THROW(std::ignore = make_mamba2(element::f32,
                                              Shape{4, 1},
                                              Shape{2, 5, 4},
                                              Shape{2, 5, 2, 16},
                                              Shape{2, 5, 4, 8},
                                              Shape{2, 5, 2, 16},
                                              Shape{2, 4, 8, 16}),
                    NodeValidationFailure,
                    testing::HasSubstr("Rank of `A` input should be in [1] list"));
}

TEST(type_prop, mamba2_invalid_C_rank) {
    OV_EXPECT_THROW(std::ignore = make_mamba2(element::f32,
                                              Shape{4},
                                              Shape{2, 5, 4},
                                              Shape{2, 5, 2, 16},
                                              Shape{2, 5, 4, 8},
                                              Shape{2, 5, 2, 16, 1},
                                              Shape{2, 4, 8, 16}),
                    NodeValidationFailure,
                    testing::HasSubstr("Rank of `C` input should be in [4] list"));
}

TEST(type_prop, mamba2_invalid_type) {
    OV_EXPECT_THROW(std::ignore = make_mamba2(element::i32,
                                              Shape{4},
                                              Shape{2, 5, 4},
                                              Shape{2, 5, 2, 16},
                                              Shape{2, 5, 4, 8},
                                              Shape{2, 5, 2, 16},
                                              Shape{2, 4, 8, 16}),
                    NodeValidationFailure,
                    testing::HasSubstr("Element type of `A` input should be in"));
}

TEST(type_prop, mamba2_state_size_mismatch) {
    OV_EXPECT_THROW(std::ignore = make_mamba2(element::f32,
                                              Shape{4},
                                              Shape{2, 5, 4},
                                              Shape{2, 5, 2, 16},
                                              Shape{2, 5, 4, 8},
                                              Shape{2, 5, 2, 32},
                                              Shape{2, 4, 8, 16}),
                    NodeValidationFailure,
                    testing::HasSubstr("The state size of `B`, `C` and `recurrent_state` should be the same."));
}

TEST(type_prop, mamba2_head_dim_mismatch) {
    OV_EXPECT_THROW(std::ignore = make_mamba2(element::f32,
                                              Shape{4},
                                              Shape{2, 5, 4},
                                              Shape{2, 5, 2, 16},
                                              Shape{2, 5, 4, 8},
                                              Shape{2, 5, 2, 16},
                                              Shape{2, 4, 10, 16}),
                    NodeValidationFailure,
                    testing::HasSubstr("The head dimension of `x` and `recurrent_state` should be the same."));
}

TEST(type_prop, mamba2_heads_not_divisible_by_groups) {
    OV_EXPECT_THROW(std::ignore = make_mamba2(element::f32,
                                              Shape{4},
                                              Shape{2, 5, 4},
                                              Shape{2, 5, 3, 16},
                                              Shape{2, 5, 4, 8},
                                              Shape{2, 5, 3, 16},
                                              Shape{2, 4, 8, 16}),
                    NodeValidationFailure,
                    testing::HasSubstr("The number of heads should be divisible by the number of groups."));
}

}  // namespace ov::test
