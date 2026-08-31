// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_selective_ssm.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/openvino.hpp"

namespace ov::test {
namespace {

std::shared_ptr<op::internal::PagedSelectiveSSM> make_paged_selective_ssm(const element::Type& et,
                                                                          const element::Type& state_et,
                                                                          const element::Type& ind_et,
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
    auto state_p = std::make_shared<op::v0::Parameter>(state_et, state);
    auto subseq = std::make_shared<op::v0::Parameter>(ind_et, PartialShape{-1});
    auto block_idx = std::make_shared<op::v0::Parameter>(ind_et, PartialShape{-1});
    auto block_idx_begins = std::make_shared<op::v0::Parameter>(ind_et, PartialShape{-1});
    auto processed = std::make_shared<op::v0::Parameter>(ind_et, PartialShape{-1});
    auto cache_interval = std::make_shared<op::v0::Parameter>(ind_et, PartialShape{-1});

    return std::make_shared<op::internal::PagedSelectiveSSM>(OutputVector{A_p,
                                                                          dt_p,
                                                                          B_p,
                                                                          x_p,
                                                                          C_p,
                                                                          state_p,
                                                                          subseq,
                                                                          block_idx,
                                                                          block_idx_begins,
                                                                          processed,
                                                                          cache_interval});
}

std::shared_ptr<op::internal::PagedSelectiveSSM> make_paged_selective_ssm(const element::Type& et,
                                                                          const element::Type& ind_et,
                                                                          const PartialShape& A,
                                                                          const PartialShape& dt,
                                                                          const PartialShape& B,
                                                                          const PartialShape& x,
                                                                          const PartialShape& C,
                                                                          const PartialShape& state) {
    return make_paged_selective_ssm(et, et, ind_et, A, dt, B, x, C, state);
}

}  // namespace

TEST(type_prop, paged_selective_ssm_static_f32) {
    const auto op = make_paged_selective_ssm(element::f32,
                                             element::i32,
                                             Shape{4},
                                             Shape{6, 4},
                                             Shape{6, 2, 16},
                                             Shape{6, 4, 8},
                                             Shape{6, 2, 16},
                                             Shape{3, 4, 8, 16});

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape(Shape{6, 4, 8}));
}

TEST(type_prop, paged_selective_ssm_partial_shape_infer) {
    const auto op = make_paged_selective_ssm(element::f32,
                                             element::i32,
                                             Shape{4},
                                             Shape{6, 4},
                                             Shape{6, 2, 16},
                                             PartialShape{-1, -1, 8},
                                             Shape{6, 2, 16},
                                             Shape{3, 4, 8, 16});

    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape(Shape{6, 4, 8}));
}

TEST(type_prop, paged_selective_ssm_state_type_is_independent) {
    const auto check_types =
        [](const element::Type& data_type, const element::Type& state_type, const element::Type& index_type) {
            const auto op = make_paged_selective_ssm(data_type,
                                                     state_type,
                                                     index_type,
                                                     Shape{4},
                                                     Shape{6, 4},
                                                     Shape{6, 2, 16},
                                                     Shape{6, 4, 8},
                                                     Shape{6, 2, 16},
                                                     Shape{3, 4, 8, 16});
            EXPECT_EQ(op->get_output_element_type(0), data_type);
        };

    check_types(element::f16, element::bf16, element::i64);
    check_types(element::f32, element::f16, element::i32);
    check_types(element::f16, element::f32, element::i32);
    check_types(element::f32, element::dynamic, element::i32);
}

TEST(type_prop, paged_selective_ssm_merges_dynamic_computation_input_type) {
    auto A = std::make_shared<op::v0::Parameter>(element::dynamic, Shape{4});
    auto dt = std::make_shared<op::v0::Parameter>(element::f32, Shape{6, 4});
    auto B = std::make_shared<op::v0::Parameter>(element::f32, Shape{6, 2, 16});
    auto x = std::make_shared<op::v0::Parameter>(element::f32, Shape{6, 4, 8});
    auto C = std::make_shared<op::v0::Parameter>(element::f32, Shape{6, 2, 16});
    auto state = std::make_shared<op::v0::Parameter>(element::f16, Shape{3, 4, 8, 16});
    auto subseq = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    auto block_idx = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    auto block_idx_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    auto processed = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});

    const auto op = std::make_shared<op::internal::PagedSelectiveSSM>(
        OutputVector{A, dt, B, x, C, state, subseq, block_idx, block_idx_begins, processed, cache_interval});

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
}

TEST(type_prop, paged_selective_ssm_bad_index_type) {
    OV_EXPECT_THROW(std::ignore = make_paged_selective_ssm(element::f32,
                                                           element::f32,
                                                           Shape{4},
                                                           Shape{6, 4},
                                                           Shape{6, 2, 16},
                                                           Shape{6, 4, 8},
                                                           Shape{6, 2, 16},
                                                           Shape{3, 4, 8, 16}),
                    NodeValidationFailure,
                    testing::HasSubstr("metadata inputs must have i32 or i64 element type."));
}

TEST(type_prop, paged_selective_ssm_f16_and_bf16_accepted) {
    for (const auto& et : {element::f16, element::bf16}) {
        const auto op = make_paged_selective_ssm(et,
                                                 element::i32,
                                                 Shape{4},
                                                 Shape{6, 4},
                                                 Shape{6, 2, 16},
                                                 Shape{6, 4, 8},
                                                 Shape{6, 2, 16},
                                                 Shape{3, 4, 8, 16});

        EXPECT_EQ(op->get_output_element_type(0), et);
    }
}

TEST(type_prop, paged_selective_ssm_unsupported_float_type) {
    OV_EXPECT_THROW(std::ignore = make_paged_selective_ssm(element::f64,
                                                           element::i32,
                                                           Shape{4},
                                                           Shape{6, 4},
                                                           Shape{6, 2, 16},
                                                           Shape{6, 4, 8},
                                                           Shape{6, 2, 16},
                                                           Shape{3, 4, 8, 16}),
                    NodeValidationFailure,
                    testing::HasSubstr("PagedSelectiveSSM data inputs must have f32, f16, or bf16 element type."));
}

TEST(type_prop, paged_selective_ssm_heads_not_divisible_by_groups) {
    OV_EXPECT_THROW(std::ignore = make_paged_selective_ssm(element::f32,
                                                           element::i32,
                                                           Shape{4},
                                                           Shape{6, 4},
                                                           Shape{6, 3, 16},
                                                           Shape{6, 4, 8},
                                                           Shape{6, 3, 16},
                                                           Shape{3, 4, 8, 16}),
                    NodeValidationFailure,
                    testing::HasSubstr("The number of heads should be divisible by the number of groups."));
}

TEST(type_prop, paged_selective_ssm_wrong_input_count) {
    auto A = std::make_shared<op::v0::Parameter>(element::f32, Shape{4});
    auto dt = std::make_shared<op::v0::Parameter>(element::f32, Shape{6, 4});
    auto B = std::make_shared<op::v0::Parameter>(element::f32, Shape{6, 2, 16});
    auto x = std::make_shared<op::v0::Parameter>(element::f32, Shape{6, 4, 8});
    auto C = std::make_shared<op::v0::Parameter>(element::f32, Shape{6, 2, 16});
    auto state = std::make_shared<op::v0::Parameter>(element::f32, Shape{3, 4, 8, 16});
    auto subseq = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    auto block_idx = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    auto block_idx_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    auto processed = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});

    EXPECT_THROW(std::ignore = std::make_shared<op::internal::PagedSelectiveSSM>(
                     OutputVector{A, dt, B, x, C, state, subseq, block_idx, block_idx_begins, processed}),
                 NodeValidationFailure);
}

TEST(type_prop, paged_selective_ssm_invalid_A_rank) {
    EXPECT_THROW(std::ignore = make_paged_selective_ssm(element::f32,
                                                        element::i32,
                                                        Shape{4, 1},
                                                        Shape{6, 4},
                                                        Shape{6, 2, 16},
                                                        Shape{6, 4, 8},
                                                        Shape{6, 2, 16},
                                                        Shape{3, 4, 8, 16}),
                 NodeValidationFailure);
}

TEST(type_prop, paged_selective_ssm_invalid_dt_rank) {
    EXPECT_THROW(std::ignore = make_paged_selective_ssm(element::f32,
                                                        element::i32,
                                                        Shape{4},
                                                        Shape{6, 4, 1},
                                                        Shape{6, 2, 16},
                                                        Shape{6, 4, 8},
                                                        Shape{6, 2, 16},
                                                        Shape{3, 4, 8, 16}),
                 NodeValidationFailure);
}

TEST(type_prop, paged_selective_ssm_invalid_B_rank) {
    EXPECT_THROW(std::ignore = make_paged_selective_ssm(element::f32,
                                                        element::i32,
                                                        Shape{4},
                                                        Shape{6, 4},
                                                        Shape{6, 2},
                                                        Shape{6, 4, 8},
                                                        Shape{6, 2, 16},
                                                        Shape{3, 4, 8, 16}),
                 NodeValidationFailure);
}

TEST(type_prop, paged_selective_ssm_invalid_x_rank) {
    EXPECT_THROW(std::ignore = make_paged_selective_ssm(element::f32,
                                                        element::i32,
                                                        Shape{4},
                                                        Shape{6, 4},
                                                        Shape{6, 2, 16},
                                                        Shape{6, 4},
                                                        Shape{6, 2, 16},
                                                        Shape{3, 4, 8, 16}),
                 NodeValidationFailure);
}

TEST(type_prop, paged_selective_ssm_invalid_C_rank) {
    EXPECT_THROW(std::ignore = make_paged_selective_ssm(element::f32,
                                                        element::i32,
                                                        Shape{4},
                                                        Shape{6, 4},
                                                        Shape{6, 2, 16},
                                                        Shape{6, 4, 8},
                                                        Shape{6, 2},
                                                        Shape{3, 4, 8, 16}),
                 NodeValidationFailure);
}

TEST(type_prop, paged_selective_ssm_invalid_state_rank) {
    EXPECT_THROW(std::ignore = make_paged_selective_ssm(element::f32,
                                                        element::i32,
                                                        Shape{4},
                                                        Shape{6, 4},
                                                        Shape{6, 2, 16},
                                                        Shape{6, 4, 8},
                                                        Shape{6, 2, 16},
                                                        Shape{3, 4, 8}),
                 NodeValidationFailure);
}

TEST(type_prop, paged_selective_ssm_invalid_index_rank) {
    auto A = std::make_shared<op::v0::Parameter>(element::f32, Shape{4});
    auto dt = std::make_shared<op::v0::Parameter>(element::f32, Shape{6, 4});
    auto B = std::make_shared<op::v0::Parameter>(element::f32, Shape{6, 2, 16});
    auto x = std::make_shared<op::v0::Parameter>(element::f32, Shape{6, 4, 8});
    auto C = std::make_shared<op::v0::Parameter>(element::f32, Shape{6, 2, 16});
    auto state = std::make_shared<op::v0::Parameter>(element::f32, Shape{3, 4, 8, 16});
    auto subseq = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1, -1});
    auto block_idx = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    auto block_idx_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    auto processed = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});

    EXPECT_THROW(
        std::ignore = std::make_shared<op::internal::PagedSelectiveSSM>(
            OutputVector{A, dt, B, x, C, state, subseq, block_idx, block_idx_begins, processed, cache_interval}),
        NodeValidationFailure);
}

TEST(type_prop, paged_selective_ssm_batch_tokens_mismatch) {
    OV_EXPECT_THROW(std::ignore = make_paged_selective_ssm(element::f32,
                                                           element::i32,
                                                           Shape{4},
                                                           Shape{7, 4},
                                                           Shape{6, 2, 16},
                                                           Shape{6, 4, 8},
                                                           Shape{6, 2, 16},
                                                           Shape{3, 4, 8, 16}),
                    NodeValidationFailure,
                    testing::HasSubstr("The token dimension of `dt`, `B`, `x` and `C` should be the same."));
}

TEST(type_prop, paged_selective_ssm_num_heads_mismatch) {
    OV_EXPECT_THROW(
        std::ignore = make_paged_selective_ssm(element::f32,
                                               element::i32,
                                               Shape{5},
                                               Shape{6, 4},
                                               Shape{6, 2, 16},
                                               Shape{6, 4, 8},
                                               Shape{6, 2, 16},
                                               Shape{3, 4, 8, 16}),
        NodeValidationFailure,
        testing::HasSubstr("The number of heads of `A`, `dt`, `x` and `recurrent_state_table` should be the same."));
}

TEST(type_prop, paged_selective_ssm_num_groups_mismatch) {
    OV_EXPECT_THROW(std::ignore = make_paged_selective_ssm(element::f32,
                                                           element::i32,
                                                           Shape{4},
                                                           Shape{6, 4},
                                                           Shape{6, 2, 16},
                                                           Shape{6, 4, 8},
                                                           Shape{6, 3, 16},
                                                           Shape{3, 4, 8, 16}),
                    NodeValidationFailure,
                    testing::HasSubstr("The number of groups of `B` and `C` should be the same."));
}

TEST(type_prop, paged_selective_ssm_head_dim_mismatch) {
    OV_EXPECT_THROW(std::ignore = make_paged_selective_ssm(element::f32,
                                                           element::i32,
                                                           Shape{4},
                                                           Shape{6, 4},
                                                           Shape{6, 2, 16},
                                                           Shape{6, 4, 8},
                                                           Shape{6, 2, 16},
                                                           Shape{3, 4, 10, 16}),
                    NodeValidationFailure,
                    testing::HasSubstr("The head dimension of `x` and `recurrent_state_table` should be the same."));
}

TEST(type_prop, paged_selective_ssm_state_size_mismatch) {
    OV_EXPECT_THROW(std::ignore = make_paged_selective_ssm(element::f32,
                                                           element::i32,
                                                           Shape{4},
                                                           Shape{6, 4},
                                                           Shape{6, 2, 16},
                                                           Shape{6, 4, 8},
                                                           Shape{6, 2, 32},
                                                           Shape{3, 4, 8, 16}),
                    NodeValidationFailure,
                    testing::HasSubstr("The state size of `B`, `C` and `recurrent_state_table` should be the same."));
}

TEST(type_prop, paged_selective_ssm_logical_and_physical_block_counts_are_independent) {
    auto A = std::make_shared<op::v0::Parameter>(element::f32, Shape{4});
    auto dt = std::make_shared<op::v0::Parameter>(element::f32, Shape{6, 4});
    auto B = std::make_shared<op::v0::Parameter>(element::f32, Shape{6, 2, 16});
    auto x = std::make_shared<op::v0::Parameter>(element::f32, Shape{6, 4, 8});
    auto C = std::make_shared<op::v0::Parameter>(element::f32, Shape{6, 2, 16});
    auto state = std::make_shared<op::v0::Parameter>(element::f32, Shape{3, 4, 8, 16});
    auto subseq = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    auto block_idx = std::make_shared<op::v0::Parameter>(element::i32, Shape{5});
    auto block_idx_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    auto processed = std::make_shared<op::v0::Parameter>(element::i32, Shape{1});
    auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, Shape{1});

    const auto op = std::make_shared<op::internal::PagedSelectiveSSM>(
        OutputVector{A, dt, B, x, C, state, subseq, block_idx, block_idx_begins, processed, cache_interval});
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape(Shape{6, 4, 8}));
}

TEST(type_prop, paged_selective_ssm_over_provisioned_state_table) {
    // The table may carry more physical rows than the logical slots addressing it.
    auto A = std::make_shared<op::v0::Parameter>(element::f32, Shape{4});
    auto dt = std::make_shared<op::v0::Parameter>(element::f32, Shape{6, 4});
    auto B = std::make_shared<op::v0::Parameter>(element::f32, Shape{6, 2, 16});
    auto x = std::make_shared<op::v0::Parameter>(element::f32, Shape{6, 4, 8});
    auto C = std::make_shared<op::v0::Parameter>(element::f32, Shape{6, 2, 16});
    auto state = std::make_shared<op::v0::Parameter>(element::f32, Shape{12, 4, 8, 16});
    auto subseq = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    auto block_idx = std::make_shared<op::v0::Parameter>(element::i32, Shape{7});
    auto block_idx_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    auto processed = std::make_shared<op::v0::Parameter>(element::i32, Shape{1});
    auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, Shape{1});

    const auto op = std::make_shared<op::internal::PagedSelectiveSSM>(
        OutputVector{A, dt, B, x, C, state, subseq, block_idx, block_idx_begins, processed, cache_interval});
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape(Shape{6, 4, 8}));
}

TEST(type_prop, paged_selective_ssm_subsequence_and_block_begins_mismatch) {
    auto A = std::make_shared<op::v0::Parameter>(element::f32, Shape{4});
    auto dt = std::make_shared<op::v0::Parameter>(element::f32, Shape{6, 4});
    auto B = std::make_shared<op::v0::Parameter>(element::f32, Shape{6, 2, 16});
    auto x = std::make_shared<op::v0::Parameter>(element::f32, Shape{6, 4, 8});
    auto C = std::make_shared<op::v0::Parameter>(element::f32, Shape{6, 2, 16});
    auto state = std::make_shared<op::v0::Parameter>(element::f32, Shape{3, 4, 8, 16});
    auto subseq = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    auto block_idx = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    auto block_idx_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    auto processed = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});

    OV_EXPECT_THROW(
        std::ignore = std::make_shared<op::internal::PagedSelectiveSSM>(
            OutputVector{A, dt, B, x, C, state, subseq, block_idx, block_idx_begins, processed, cache_interval}),
        NodeValidationFailure,
        testing::HasSubstr("The sizes of `subsequence_begins` and `la_block_indices_begins` should be the same."));
}

TEST(type_prop, paged_selective_ssm_processed_tokens_and_cache_interval_mismatch) {
    auto A = std::make_shared<op::v0::Parameter>(element::f32, Shape{4});
    auto dt = std::make_shared<op::v0::Parameter>(element::f32, Shape{6, 4});
    auto B = std::make_shared<op::v0::Parameter>(element::f32, Shape{6, 2, 16});
    auto x = std::make_shared<op::v0::Parameter>(element::f32, Shape{6, 4, 8});
    auto C = std::make_shared<op::v0::Parameter>(element::f32, Shape{6, 2, 16});
    auto state = std::make_shared<op::v0::Parameter>(element::f32, Shape{3, 4, 8, 16});
    auto subseq = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    auto block_idx = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    auto block_idx_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    auto processed = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});

    OV_EXPECT_THROW(
        std::ignore = std::make_shared<op::internal::PagedSelectiveSSM>(
            OutputVector{A, dt, B, x, C, state, subseq, block_idx, block_idx_begins, processed, cache_interval}),
        NodeValidationFailure,
        testing::HasSubstr("The sizes of `num_processed_tokens` and `cache_interval` should be the same."));
}

TEST(type_prop, paged_selective_ssm_subsequence_begins_and_processed_tokens_mismatch) {
    auto A = std::make_shared<op::v0::Parameter>(element::f32, Shape{4});
    auto dt = std::make_shared<op::v0::Parameter>(element::f32, Shape{6, 4});
    auto B = std::make_shared<op::v0::Parameter>(element::f32, Shape{6, 2, 16});
    auto x = std::make_shared<op::v0::Parameter>(element::f32, Shape{6, 4, 8});
    auto C = std::make_shared<op::v0::Parameter>(element::f32, Shape{6, 2, 16});
    auto state = std::make_shared<op::v0::Parameter>(element::f32, Shape{3, 4, 8, 16});
    auto subseq = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    auto block_idx = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    auto block_idx_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    auto processed = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});

    OV_EXPECT_THROW(
        std::ignore = std::make_shared<op::internal::PagedSelectiveSSM>(
            OutputVector{A, dt, B, x, C, state, subseq, block_idx, block_idx_begins, processed, cache_interval}),
        NodeValidationFailure,
        testing::HasSubstr("The size of `subsequence_begins` should be one larger than `num_processed_tokens`."));
}

TEST(type_prop, paged_selective_ssm_zero_groups) {
    OV_EXPECT_THROW(std::ignore = make_paged_selective_ssm(element::f32,
                                                           element::i32,
                                                           Shape{4},
                                                           Shape{6, 4},
                                                           Shape{6, 0, 16},
                                                           Shape{6, 4, 8},
                                                           Shape{6, 0, 16},
                                                           Shape{3, 4, 8, 16}),
                    NodeValidationFailure,
                    testing::HasSubstr("The number of groups must be greater than zero."));
}

TEST(type_prop, paged_selective_ssm_dynamic_rank_input_accepted) {
    // Dynamic rank inputs are accepted since they may be folded to a static rank later.
    const auto op = make_paged_selective_ssm(element::f32,
                                             element::i32,
                                             Shape{4},
                                             Shape{6, 4},
                                             Shape{6, 2, 16},
                                             PartialShape::dynamic(),
                                             Shape{6, 2, 16},
                                             Shape{3, 4, 8, 16});

    EXPECT_TRUE(op->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop, paged_selective_ssm_dynamic_num_groups_skips_divisibility_check) {
    const auto op = make_paged_selective_ssm(element::f32,
                                             element::i32,
                                             Shape{4},
                                             Shape{6, 4},
                                             PartialShape{6, -1, 16},
                                             Shape{6, 4, 8},
                                             PartialShape{6, -1, 16},
                                             Shape{3, 4, 8, 16});

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{6, 4, 8}));
}

TEST(type_prop, paged_selective_ssm_asymmetric_num_groups_checks_divisibility) {
    // The group count is dynamic in `B` but pinned by `C`, so the merged value must drive the check.
    OV_EXPECT_THROW(std::ignore = make_paged_selective_ssm(element::f32,
                                                           element::i32,
                                                           Shape{4},
                                                           Shape{6, 4},
                                                           PartialShape{6, -1, 16},
                                                           Shape{6, 4, 8},
                                                           Shape{6, 3, 16},
                                                           Shape{3, 4, 8, 16}),
                    NodeValidationFailure,
                    testing::HasSubstr("The number of heads should be divisible by the number of groups."));
}

TEST(type_prop, paged_selective_ssm_asymmetric_num_groups_accepted) {
    const auto op = make_paged_selective_ssm(element::f32,
                                             element::i32,
                                             Shape{4},
                                             Shape{6, 4},
                                             PartialShape{6, -1, 16},
                                             Shape{6, 4, 8},
                                             Shape{6, 2, 16},
                                             Shape{3, 4, 8, 16});

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{6, 4, 8}));
}

TEST(type_prop, paged_selective_ssm_type_mismatch) {
    auto A = std::make_shared<op::v0::Parameter>(element::f16, Shape{4});
    auto dt = std::make_shared<op::v0::Parameter>(element::f32, Shape{6, 4});
    auto B = std::make_shared<op::v0::Parameter>(element::f32, Shape{6, 2, 16});
    auto x = std::make_shared<op::v0::Parameter>(element::f32, Shape{6, 4, 8});
    auto C = std::make_shared<op::v0::Parameter>(element::f32, Shape{6, 2, 16});
    auto state = std::make_shared<op::v0::Parameter>(element::f32, Shape{3, 4, 8, 16});
    auto subseq = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    auto block_idx = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    auto block_idx_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    auto processed = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});

    OV_EXPECT_THROW(
        std::ignore = std::make_shared<op::internal::PagedSelectiveSSM>(
            OutputVector{A, dt, B, x, C, state, subseq, block_idx, block_idx_begins, processed, cache_interval}),
        NodeValidationFailure,
        testing::HasSubstr("PagedSelectiveSSM expects inputs A, dt, B, x, and C to have the same element type."));
}

TEST(type_prop, paged_selective_ssm_state_float_type_invalid) {
    auto A = std::make_shared<op::v0::Parameter>(element::f32, Shape{4});
    auto dt = std::make_shared<op::v0::Parameter>(element::f32, Shape{6, 4});
    auto B = std::make_shared<op::v0::Parameter>(element::f32, Shape{6, 2, 16});
    auto x = std::make_shared<op::v0::Parameter>(element::f32, Shape{6, 4, 8});
    auto C = std::make_shared<op::v0::Parameter>(element::f32, Shape{6, 2, 16});
    auto state = std::make_shared<op::v0::Parameter>(element::i32, Shape{3, 4, 8, 16});
    auto subseq = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    auto block_idx = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    auto block_idx_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    auto processed = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});

    OV_EXPECT_THROW(
        std::ignore = std::make_shared<op::internal::PagedSelectiveSSM>(
            OutputVector{A, dt, B, x, C, state, subseq, block_idx, block_idx_begins, processed, cache_interval}),
        NodeValidationFailure,
        testing::HasSubstr("PagedSelectiveSSM recurrent_state_table must have f32, f16, or bf16 element type."));
}

TEST(type_prop, paged_selective_ssm_index_type_mixed) {
    auto A = std::make_shared<op::v0::Parameter>(element::f32, Shape{4});
    auto dt = std::make_shared<op::v0::Parameter>(element::f32, Shape{6, 4});
    auto B = std::make_shared<op::v0::Parameter>(element::f32, Shape{6, 2, 16});
    auto x = std::make_shared<op::v0::Parameter>(element::f32, Shape{6, 4, 8});
    auto C = std::make_shared<op::v0::Parameter>(element::f32, Shape{6, 2, 16});
    auto state = std::make_shared<op::v0::Parameter>(element::f32, Shape{3, 4, 8, 16});
    auto subseq = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    auto block_idx = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    auto block_idx_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    auto processed = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    auto cache_interval = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1});

    OV_EXPECT_THROW(
        std::ignore = std::make_shared<op::internal::PagedSelectiveSSM>(
            OutputVector{A, dt, B, x, C, state, subseq, block_idx, block_idx_begins, processed, cache_interval}),
        NodeValidationFailure,
        testing::HasSubstr("expects all metadata inputs to have the same element type."));
}

}  // namespace ov::test
