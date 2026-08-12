// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/selective_ssm.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/openvino.hpp"

namespace ov::test {
namespace {

std::shared_ptr<op::internal::SelectiveSSM> make_selective_ssm(const element::Type& et,
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

    return std::make_shared<op::internal::SelectiveSSM>(OutputVector{A_p, dt_p, B_p, x_p, C_p, recurrent_state});
}

}  // namespace

TEST(type_prop, selective_ssm_static_f32) {
    const auto op = make_selective_ssm(element::f32,
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

TEST(type_prop, selective_ssm_partial_shape_infer) {
    const auto op = make_selective_ssm(element::bf16,
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

TEST(type_prop, selective_ssm_dims_resolved_from_other_inputs) {
    const auto op = make_selective_ssm(element::f32,
                                       Shape{4},
                                       Shape{2, 5, 4},
                                       Shape{2, 5, 2, 16},
                                       PartialShape{-1, -1, -1, -1},
                                       Shape{2, 5, 2, 16},
                                       Shape{2, 4, 8, 16});

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 5, 4, 8}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{2, 4, 8, 16}));
}

TEST(type_prop, selective_ssm_invalid_A_rank) {
    EXPECT_THROW(std::ignore = make_selective_ssm(element::f32,
                                                  Shape{4, 1},
                                                  Shape{2, 5, 4},
                                                  Shape{2, 5, 2, 16},
                                                  Shape{2, 5, 4, 8},
                                                  Shape{2, 5, 2, 16},
                                                  Shape{2, 4, 8, 16}),
                 NodeValidationFailure);
}

TEST(type_prop, selective_ssm_state_size_mismatch) {
    OV_EXPECT_THROW(std::ignore = make_selective_ssm(element::f32,
                                                     Shape{4},
                                                     Shape{2, 5, 4},
                                                     Shape{2, 5, 2, 16},
                                                     Shape{2, 5, 4, 8},
                                                     Shape{2, 5, 2, 32},
                                                     Shape{2, 4, 8, 16}),
                    NodeValidationFailure,
                    testing::HasSubstr("The state size of `B`, `C` and `recurrent_state` should be the same."));
}

TEST(type_prop, selective_ssm_head_dim_mismatch) {
    OV_EXPECT_THROW(std::ignore = make_selective_ssm(element::f32,
                                                     Shape{4},
                                                     Shape{2, 5, 4},
                                                     Shape{2, 5, 2, 16},
                                                     Shape{2, 5, 4, 8},
                                                     Shape{2, 5, 2, 16},
                                                     Shape{2, 4, 10, 16}),
                    NodeValidationFailure,
                    testing::HasSubstr("The head dimension of `x` and `recurrent_state` should be the same."));
}

TEST(type_prop, selective_ssm_heads_not_divisible_by_groups) {
    OV_EXPECT_THROW(std::ignore = make_selective_ssm(element::f32,
                                                     Shape{4},
                                                     Shape{2, 5, 4},
                                                     Shape{2, 5, 3, 16},
                                                     Shape{2, 5, 4, 8},
                                                     Shape{2, 5, 3, 16},
                                                     Shape{2, 4, 8, 16}),
                    NodeValidationFailure,
                    testing::HasSubstr("The number of heads should be divisible by the number of groups."));
}

TEST(type_prop, selective_ssm_type_mismatch) {
    auto A = std::make_shared<op::v0::Parameter>(element::f16, Shape{4});
    auto dt = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 5, 4});
    auto B = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 5, 2, 16});
    auto x = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 5, 4, 8});
    auto C = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 5, 2, 16});
    auto recurrent_state = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 4, 8, 16});

    OV_EXPECT_THROW(
        std::ignore = std::make_shared<op::internal::SelectiveSSM>(OutputVector{A, dt, B, x, C, recurrent_state}),
        NodeValidationFailure,
        testing::HasSubstr("SelectiveSSM expects all inputs to have the same element type."));
}

TEST(type_prop, selective_ssm_state_type_must_match) {
    auto A = std::make_shared<op::v0::Parameter>(element::f16, Shape{4});
    auto dt = std::make_shared<op::v0::Parameter>(element::f16, Shape{2, 5, 4});
    auto B = std::make_shared<op::v0::Parameter>(element::f16, Shape{2, 5, 2, 16});
    auto x = std::make_shared<op::v0::Parameter>(element::f16, Shape{2, 5, 4, 8});
    auto C = std::make_shared<op::v0::Parameter>(element::f16, Shape{2, 5, 2, 16});
    auto recurrent_state = std::make_shared<op::v0::Parameter>(element::bf16, Shape{2, 4, 8, 16});

    OV_EXPECT_THROW(
        std::ignore = std::make_shared<op::internal::SelectiveSSM>(OutputVector{A, dt, B, x, C, recurrent_state}),
        NodeValidationFailure,
        testing::HasSubstr("SelectiveSSM expects all inputs to have the same element type."));
}

TEST(type_prop, selective_ssm_f16_and_bf16_accepted) {
    for (const auto& et : {element::f16, element::bf16}) {
        const auto op = make_selective_ssm(et,
                                           Shape{4},
                                           Shape{2, 5, 4},
                                           Shape{2, 5, 2, 16},
                                           Shape{2, 5, 4, 8},
                                           Shape{2, 5, 2, 16},
                                           Shape{2, 4, 8, 16});

        EXPECT_EQ(op->get_output_element_type(0), et);
        EXPECT_EQ(op->get_output_element_type(1), et);
    }
}

TEST(type_prop, selective_ssm_unsupported_float_type) {
    OV_EXPECT_THROW(std::ignore = make_selective_ssm(element::f64,
                                                     Shape{4},
                                                     Shape{2, 5, 4},
                                                     Shape{2, 5, 2, 16},
                                                     Shape{2, 5, 4, 8},
                                                     Shape{2, 5, 2, 16},
                                                     Shape{2, 4, 8, 16}),
                    NodeValidationFailure,
                    testing::HasSubstr("SelectiveSSM inputs must have f32, f16, or bf16 element type."));
}

TEST(type_prop, selective_ssm_wrong_input_count) {
    auto A = std::make_shared<op::v0::Parameter>(element::f32, Shape{4});
    auto dt = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 5, 4});
    auto B = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 5, 2, 16});
    auto x = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 5, 4, 8});
    auto C = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 5, 2, 16});

    OV_EXPECT_THROW(std::ignore = std::make_shared<op::internal::SelectiveSSM>(OutputVector{A, dt, B, x, C}),
                    NodeValidationFailure,
                    testing::HasSubstr("SelectiveSSM expects 6 inputs, but it has 5"));
}

TEST(type_prop, selective_ssm_invalid_dt_rank) {
    EXPECT_THROW(std::ignore = make_selective_ssm(element::f32,
                                                  Shape{4},
                                                  Shape{2, 5, 4, 1},
                                                  Shape{2, 5, 2, 16},
                                                  Shape{2, 5, 4, 8},
                                                  Shape{2, 5, 2, 16},
                                                  Shape{2, 4, 8, 16}),
                 NodeValidationFailure);
}

TEST(type_prop, selective_ssm_invalid_B_rank) {
    EXPECT_THROW(std::ignore = make_selective_ssm(element::f32,
                                                  Shape{4},
                                                  Shape{2, 5, 4},
                                                  Shape{2, 5, 2},
                                                  Shape{2, 5, 4, 8},
                                                  Shape{2, 5, 2, 16},
                                                  Shape{2, 4, 8, 16}),
                 NodeValidationFailure);
}

TEST(type_prop, selective_ssm_invalid_x_rank) {
    EXPECT_THROW(std::ignore = make_selective_ssm(element::f32,
                                                  Shape{4},
                                                  Shape{2, 5, 4},
                                                  Shape{2, 5, 2, 16},
                                                  Shape{2, 5, 4},
                                                  Shape{2, 5, 2, 16},
                                                  Shape{2, 4, 8, 16}),
                 NodeValidationFailure);
}

TEST(type_prop, selective_ssm_invalid_C_rank) {
    EXPECT_THROW(std::ignore = make_selective_ssm(element::f32,
                                                  Shape{4},
                                                  Shape{2, 5, 4},
                                                  Shape{2, 5, 2, 16},
                                                  Shape{2, 5, 4, 8},
                                                  Shape{2, 5, 2},
                                                  Shape{2, 4, 8, 16}),
                 NodeValidationFailure);
}

TEST(type_prop, selective_ssm_invalid_recurrent_state_rank) {
    EXPECT_THROW(std::ignore = make_selective_ssm(element::f32,
                                                  Shape{4},
                                                  Shape{2, 5, 4},
                                                  Shape{2, 5, 2, 16},
                                                  Shape{2, 5, 4, 8},
                                                  Shape{2, 5, 2, 16},
                                                  Shape{2, 4, 8}),
                 NodeValidationFailure);
}

TEST(type_prop, selective_ssm_dynamic_rank_input_accepted) {
    // Dynamic rank inputs are accepted since they may be folded to a static rank later.
    const auto op = make_selective_ssm(element::f32,
                                       Shape{4},
                                       Shape{2, 5, 4},
                                       Shape{2, 5, 2, 16},
                                       PartialShape::dynamic(),
                                       Shape{2, 5, 2, 16},
                                       Shape{2, 4, 8, 16});

    EXPECT_TRUE(op->get_output_partial_shape(0).rank().is_dynamic());
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{2, 4, 8, 16}));
}

TEST(type_prop, selective_ssm_batch_mismatch) {
    OV_EXPECT_THROW(std::ignore = make_selective_ssm(element::f32,
                                                     Shape{4},
                                                     Shape{3, 5, 4},
                                                     Shape{2, 5, 2, 16},
                                                     Shape{2, 5, 4, 8},
                                                     Shape{2, 5, 2, 16},
                                                     Shape{2, 4, 8, 16}),
                    NodeValidationFailure,
                    testing::HasSubstr("The batch dimension of `dt`, `B`, `x`, `C` and `recurrent_state` should be "
                                       "the same."));
}

TEST(type_prop, selective_ssm_seq_len_mismatch) {
    OV_EXPECT_THROW(std::ignore = make_selective_ssm(element::f32,
                                                     Shape{4},
                                                     Shape{2, 7, 4},
                                                     Shape{2, 5, 2, 16},
                                                     Shape{2, 5, 4, 8},
                                                     Shape{2, 5, 2, 16},
                                                     Shape{2, 4, 8, 16}),
                    NodeValidationFailure,
                    testing::HasSubstr("The sequence length of `dt`, `B`, `x` and `C` should be the same."));
}

TEST(type_prop, selective_ssm_num_heads_mismatch) {
    OV_EXPECT_THROW(
        std::ignore = make_selective_ssm(element::f32,
                                         Shape{5},
                                         Shape{2, 5, 4},
                                         Shape{2, 5, 2, 16},
                                         Shape{2, 5, 4, 8},
                                         Shape{2, 5, 2, 16},
                                         Shape{2, 4, 8, 16}),
        NodeValidationFailure,
        testing::HasSubstr("The number of heads of `A`, `dt`, `x` and `recurrent_state` should be the same."));
}

TEST(type_prop, selective_ssm_num_groups_mismatch) {
    OV_EXPECT_THROW(std::ignore = make_selective_ssm(element::f32,
                                                     Shape{4},
                                                     Shape{2, 5, 4},
                                                     Shape{2, 5, 2, 16},
                                                     Shape{2, 5, 4, 8},
                                                     Shape{2, 5, 3, 16},
                                                     Shape{2, 4, 8, 16}),
                    NodeValidationFailure,
                    testing::HasSubstr("The number of groups of `B` and `C` should be the same."));
}

TEST(type_prop, selective_ssm_zero_groups) {
    OV_EXPECT_THROW(std::ignore = make_selective_ssm(element::f32,
                                                     Shape{4},
                                                     Shape{2, 5, 4},
                                                     Shape{2, 5, 0, 16},
                                                     Shape{2, 5, 4, 8},
                                                     Shape{2, 5, 0, 16},
                                                     Shape{2, 4, 8, 16}),
                    NodeValidationFailure,
                    testing::HasSubstr("The number of groups must be greater than zero."));
}

TEST(type_prop, selective_ssm_dynamic_num_groups_skips_divisibility_check) {
    const auto op = make_selective_ssm(element::f32,
                                       Shape{4},
                                       Shape{2, 5, 4},
                                       PartialShape{2, 5, -1, 16},
                                       Shape{2, 5, 4, 8},
                                       PartialShape{2, 5, -1, 16},
                                       Shape{2, 4, 8, 16});

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 5, 4, 8}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{2, 4, 8, 16}));
}

TEST(type_prop, selective_ssm_asymmetric_num_groups_checks_divisibility) {
    // The group count is dynamic in `B` but pinned by `C`, so the merged value must drive the check.
    OV_EXPECT_THROW(std::ignore = make_selective_ssm(element::f32,
                                                     Shape{4},
                                                     Shape{2, 5, 4},
                                                     PartialShape{2, 5, -1, 16},
                                                     Shape{2, 5, 4, 8},
                                                     Shape{2, 5, 3, 16},
                                                     Shape{2, 4, 8, 16}),
                    NodeValidationFailure,
                    testing::HasSubstr("The number of heads should be divisible by the number of groups."));
}

TEST(type_prop, selective_ssm_asymmetric_num_groups_accepted) {
    const auto op = make_selective_ssm(element::f32,
                                       Shape{4},
                                       Shape{2, 5, 4},
                                       PartialShape{2, 5, -1, 16},
                                       Shape{2, 5, 4, 8},
                                       Shape{2, 5, 2, 16},
                                       Shape{2, 4, 8, 16});

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 5, 4, 8}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{2, 4, 8, 16}));
}

}  // namespace ov::test
