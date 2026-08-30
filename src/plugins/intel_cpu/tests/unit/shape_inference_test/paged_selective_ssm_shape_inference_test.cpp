// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_selective_ssm_shape_inference.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/op/parameter.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using ov::op::v0::Parameter;
using testing::HasSubstr;

namespace {

std::shared_ptr<op::internal::PagedSelectiveSSM> make_paged_selective_ssm() {
    const auto dynamic = PartialShape::dynamic();
    auto A = std::make_shared<Parameter>(element::f32, dynamic);
    auto dt = std::make_shared<Parameter>(element::f32, dynamic);
    auto B = std::make_shared<Parameter>(element::f32, dynamic);
    auto x = std::make_shared<Parameter>(element::f32, dynamic);
    auto C = std::make_shared<Parameter>(element::f32, dynamic);
    auto state = std::make_shared<Parameter>(element::f32, dynamic);
    auto subseq = std::make_shared<Parameter>(element::i32, dynamic);
    auto block_idx = std::make_shared<Parameter>(element::i32, dynamic);
    auto block_idx_begins = std::make_shared<Parameter>(element::i32, dynamic);
    auto processed = std::make_shared<Parameter>(element::i32, dynamic);
    auto cache_interval = std::make_shared<Parameter>(element::i32, dynamic);
    return std::make_shared<op::internal::PagedSelectiveSSM>(
        OutputVector{A, dt, B, x, C, state, subseq, block_idx, block_idx_begins, processed, cache_interval});
}

}  // namespace

TEST(StaticShapeInferenceTest, PagedSelectiveSSMBasic) {
    const auto op = make_paged_selective_ssm();

    std::vector<StaticShape> static_input_shapes = {StaticShape{4},
                                                    StaticShape{6, 4},
                                                    StaticShape{6, 2, 16},
                                                    StaticShape{6, 4, 8},
                                                    StaticShape{6, 2, 16},
                                                    StaticShape{3, 4, 8, 16},
                                                    StaticShape{2},
                                                    StaticShape{5},
                                                    StaticShape{2},
                                                    StaticShape{1},
                                                    StaticShape{1}};
    const auto static_output_shapes = shape_inference(op.get(), static_input_shapes);
    ASSERT_EQ(static_output_shapes.size(), 1);
    EXPECT_EQ(static_output_shapes[0], StaticShape({6, 4, 8}));
}

TEST(StaticShapeInferenceTest, PagedSelectiveSSMInvalidARank) {
    const auto op = make_paged_selective_ssm();

    std::vector<StaticShape> static_input_shapes = {StaticShape{4, 1},
                                                    StaticShape{6, 4},
                                                    StaticShape{6, 2, 16},
                                                    StaticShape{6, 4, 8},
                                                    StaticShape{6, 2, 16},
                                                    StaticShape{3, 4, 8, 16},
                                                    StaticShape{2},
                                                    StaticShape{5},
                                                    StaticShape{2},
                                                    StaticShape{1},
                                                    StaticShape{1}};
    EXPECT_THROW(shape_inference(op.get(), static_input_shapes), NodeValidationFailure);
}

TEST(StaticShapeInferenceTest, PagedSelectiveSSMInvalidDtRank) {
    const auto op = make_paged_selective_ssm();

    std::vector<StaticShape> static_input_shapes = {StaticShape{4},
                                                    StaticShape{6, 4, 1},
                                                    StaticShape{6, 2, 16},
                                                    StaticShape{6, 4, 8},
                                                    StaticShape{6, 2, 16},
                                                    StaticShape{3, 4, 8, 16},
                                                    StaticShape{2},
                                                    StaticShape{5},
                                                    StaticShape{2},
                                                    StaticShape{1},
                                                    StaticShape{1}};
    EXPECT_THROW(shape_inference(op.get(), static_input_shapes), NodeValidationFailure);
}

TEST(StaticShapeInferenceTest, PagedSelectiveSSMInvalidBRank) {
    const auto op = make_paged_selective_ssm();

    std::vector<StaticShape> static_input_shapes = {StaticShape{4},
                                                    StaticShape{6, 4},
                                                    StaticShape{6, 2},
                                                    StaticShape{6, 4, 8},
                                                    StaticShape{6, 2, 16},
                                                    StaticShape{3, 4, 8, 16},
                                                    StaticShape{2},
                                                    StaticShape{5},
                                                    StaticShape{2},
                                                    StaticShape{1},
                                                    StaticShape{1}};
    EXPECT_THROW(shape_inference(op.get(), static_input_shapes), NodeValidationFailure);
}

TEST(StaticShapeInferenceTest, PagedSelectiveSSMInvalidXRank) {
    const auto op = make_paged_selective_ssm();

    std::vector<StaticShape> static_input_shapes = {StaticShape{4},
                                                    StaticShape{6, 4},
                                                    StaticShape{6, 2, 16},
                                                    StaticShape{6, 4},
                                                    StaticShape{6, 2, 16},
                                                    StaticShape{3, 4, 8, 16},
                                                    StaticShape{2},
                                                    StaticShape{5},
                                                    StaticShape{2},
                                                    StaticShape{1},
                                                    StaticShape{1}};
    EXPECT_THROW(shape_inference(op.get(), static_input_shapes), NodeValidationFailure);
}

TEST(StaticShapeInferenceTest, PagedSelectiveSSMInvalidCRank) {
    const auto op = make_paged_selective_ssm();

    std::vector<StaticShape> static_input_shapes = {StaticShape{4},
                                                    StaticShape{6, 4},
                                                    StaticShape{6, 2, 16},
                                                    StaticShape{6, 4, 8},
                                                    StaticShape{6, 2},
                                                    StaticShape{3, 4, 8, 16},
                                                    StaticShape{2},
                                                    StaticShape{5},
                                                    StaticShape{2},
                                                    StaticShape{1},
                                                    StaticShape{1}};
    EXPECT_THROW(shape_inference(op.get(), static_input_shapes), NodeValidationFailure);
}

TEST(StaticShapeInferenceTest, PagedSelectiveSSMInvalidStateRank) {
    const auto op = make_paged_selective_ssm();

    std::vector<StaticShape> static_input_shapes = {StaticShape{4},
                                                    StaticShape{6, 4},
                                                    StaticShape{6, 2, 16},
                                                    StaticShape{6, 4, 8},
                                                    StaticShape{6, 2, 16},
                                                    StaticShape{3, 4, 8},
                                                    StaticShape{2},
                                                    StaticShape{5},
                                                    StaticShape{2},
                                                    StaticShape{1},
                                                    StaticShape{1}};
    EXPECT_THROW(shape_inference(op.get(), static_input_shapes), NodeValidationFailure);
}

TEST(StaticShapeInferenceTest, PagedSelectiveSSMInvalidIndexRank) {
    const auto op = make_paged_selective_ssm();

    std::vector<StaticShape> static_input_shapes = {StaticShape{4},
                                                    StaticShape{6, 4},
                                                    StaticShape{6, 2, 16},
                                                    StaticShape{6, 4, 8},
                                                    StaticShape{6, 2, 16},
                                                    StaticShape{3, 4, 8, 16},
                                                    StaticShape{2, 2},
                                                    StaticShape{5},
                                                    StaticShape{2},
                                                    StaticShape{1},
                                                    StaticShape{1}};
    EXPECT_THROW(shape_inference(op.get(), static_input_shapes), NodeValidationFailure);
}

TEST(StaticShapeInferenceTest, PagedSelectiveSSMTokenDimMismatch) {
    const auto op = make_paged_selective_ssm();

    std::vector<StaticShape> static_input_shapes = {StaticShape{4},
                                                    StaticShape{7, 4},
                                                    StaticShape{6, 2, 16},
                                                    StaticShape{6, 4, 8},
                                                    StaticShape{6, 2, 16},
                                                    StaticShape{3, 4, 8, 16},
                                                    StaticShape{2},
                                                    StaticShape{5},
                                                    StaticShape{2},
                                                    StaticShape{1},
                                                    StaticShape{1}};
    OV_EXPECT_THROW(shape_inference(op.get(), static_input_shapes),
                    NodeValidationFailure,
                    HasSubstr("The token dimension of `dt`, `B`, `x` and `C` should be the same."));
}

TEST(StaticShapeInferenceTest, PagedSelectiveSSMNumHeadsMismatch) {
    const auto op = make_paged_selective_ssm();

    std::vector<StaticShape> static_input_shapes = {StaticShape{5},
                                                    StaticShape{6, 4},
                                                    StaticShape{6, 2, 16},
                                                    StaticShape{6, 4, 8},
                                                    StaticShape{6, 2, 16},
                                                    StaticShape{3, 4, 8, 16},
                                                    StaticShape{2},
                                                    StaticShape{5},
                                                    StaticShape{2},
                                                    StaticShape{1},
                                                    StaticShape{1}};
    OV_EXPECT_THROW(shape_inference(op.get(), static_input_shapes),
                    NodeValidationFailure,
                    HasSubstr("The number of heads of `A`, `dt`, `x` and `recurrent_state_table` should be the same."));
}

TEST(StaticShapeInferenceTest, PagedSelectiveSSMHeadDimMismatch) {
    const auto op = make_paged_selective_ssm();

    std::vector<StaticShape> static_input_shapes = {StaticShape{4},
                                                    StaticShape{6, 4},
                                                    StaticShape{6, 2, 16},
                                                    StaticShape{6, 4, 8},
                                                    StaticShape{6, 2, 16},
                                                    StaticShape{3, 4, 10, 16},
                                                    StaticShape{2},
                                                    StaticShape{5},
                                                    StaticShape{2},
                                                    StaticShape{1},
                                                    StaticShape{1}};
    OV_EXPECT_THROW(shape_inference(op.get(), static_input_shapes),
                    NodeValidationFailure,
                    HasSubstr("The head dimension of `x` and `recurrent_state_table` should be the same."));
}

TEST(StaticShapeInferenceTest, PagedSelectiveSSMNumGroupsMismatch) {
    const auto op = make_paged_selective_ssm();

    std::vector<StaticShape> static_input_shapes = {StaticShape{4},
                                                    StaticShape{6, 4},
                                                    StaticShape{6, 2, 16},
                                                    StaticShape{6, 4, 8},
                                                    StaticShape{6, 3, 16},
                                                    StaticShape{3, 4, 8, 16},
                                                    StaticShape{2},
                                                    StaticShape{5},
                                                    StaticShape{2},
                                                    StaticShape{1},
                                                    StaticShape{1}};
    OV_EXPECT_THROW(shape_inference(op.get(), static_input_shapes),
                    NodeValidationFailure,
                    HasSubstr("The number of groups of `B` and `C` should be the same."));
}

TEST(StaticShapeInferenceTest, PagedSelectiveSSMZeroGroups) {
    const auto op = make_paged_selective_ssm();

    std::vector<StaticShape> static_input_shapes = {StaticShape{4},
                                                    StaticShape{6, 4},
                                                    StaticShape{6, 0, 16},
                                                    StaticShape{6, 4, 8},
                                                    StaticShape{6, 0, 16},
                                                    StaticShape{3, 4, 8, 16},
                                                    StaticShape{2},
                                                    StaticShape{5},
                                                    StaticShape{2},
                                                    StaticShape{1},
                                                    StaticShape{1}};
    OV_EXPECT_THROW(shape_inference(op.get(), static_input_shapes),
                    NodeValidationFailure,
                    HasSubstr("The number of groups must be greater than zero."));
}

TEST(StaticShapeInferenceTest, PagedSelectiveSSMStateSizeMismatch) {
    const auto op = make_paged_selective_ssm();

    std::vector<StaticShape> static_input_shapes = {StaticShape{4},
                                                    StaticShape{6, 4},
                                                    StaticShape{6, 2, 16},
                                                    StaticShape{6, 4, 8},
                                                    StaticShape{6, 2, 32},
                                                    StaticShape{3, 4, 8, 16},
                                                    StaticShape{2},
                                                    StaticShape{5},
                                                    StaticShape{2},
                                                    StaticShape{1},
                                                    StaticShape{1}};
    OV_EXPECT_THROW(shape_inference(op.get(), static_input_shapes),
                    NodeValidationFailure,
                    HasSubstr("The state size of `B`, `C` and `recurrent_state_table` should be the same."));
}

TEST(StaticShapeInferenceTest, PagedSelectiveSSMHeadsNotDivisibleByGroups) {
    const auto op = make_paged_selective_ssm();

    std::vector<StaticShape> static_input_shapes = {StaticShape{4},
                                                    StaticShape{6, 4},
                                                    StaticShape{6, 3, 16},
                                                    StaticShape{6, 4, 8},
                                                    StaticShape{6, 3, 16},
                                                    StaticShape{3, 4, 8, 16},
                                                    StaticShape{2},
                                                    StaticShape{5},
                                                    StaticShape{2},
                                                    StaticShape{1},
                                                    StaticShape{1}};
    OV_EXPECT_THROW(shape_inference(op.get(), static_input_shapes),
                    NodeValidationFailure,
                    HasSubstr("The number of heads should be divisible by the number of groups."));
}

TEST(StaticShapeInferenceTest, PagedSelectiveSSMLogicalAndPhysicalBlockCountsAreIndependent) {
    // The table may carry more logical slots (block_indices) than sequences (block_indices_begins - 1).
    const auto op = make_paged_selective_ssm();

    std::vector<StaticShape> static_input_shapes = {StaticShape{4},
                                                    StaticShape{6, 4},
                                                    StaticShape{6, 2, 16},
                                                    StaticShape{6, 4, 8},
                                                    StaticShape{6, 2, 16},
                                                    StaticShape{3, 4, 8, 16},
                                                    StaticShape{2},
                                                    StaticShape{5},
                                                    StaticShape{2},
                                                    StaticShape{1},
                                                    StaticShape{1}};
    const auto static_output_shapes = shape_inference(op.get(), static_input_shapes);
    ASSERT_EQ(static_output_shapes.size(), 1);
    EXPECT_EQ(static_output_shapes[0], StaticShape({6, 4, 8}));
}

TEST(StaticShapeInferenceTest, PagedSelectiveSSMOverProvisionedStateTable) {
    // The table may carry more physical rows than the logical slots addressing it.
    const auto op = make_paged_selective_ssm();

    std::vector<StaticShape> static_input_shapes = {StaticShape{4},
                                                    StaticShape{6, 4},
                                                    StaticShape{6, 2, 16},
                                                    StaticShape{6, 4, 8},
                                                    StaticShape{6, 2, 16},
                                                    StaticShape{12, 4, 8, 16},
                                                    StaticShape{2},
                                                    StaticShape{7},
                                                    StaticShape{2},
                                                    StaticShape{1},
                                                    StaticShape{1}};
    const auto static_output_shapes = shape_inference(op.get(), static_input_shapes);
    ASSERT_EQ(static_output_shapes.size(), 1);
    EXPECT_EQ(static_output_shapes[0], StaticShape({6, 4, 8}));
}

TEST(StaticShapeInferenceTest, PagedSelectiveSSMSubsequenceAndBlockBeginsMismatch) {
    const auto op = make_paged_selective_ssm();

    std::vector<StaticShape> static_input_shapes = {StaticShape{4},
                                                    StaticShape{6, 4},
                                                    StaticShape{6, 2, 16},
                                                    StaticShape{6, 4, 8},
                                                    StaticShape{6, 2, 16},
                                                    StaticShape{3, 4, 8, 16},
                                                    StaticShape{3},
                                                    StaticShape{3},
                                                    StaticShape{2},
                                                    StaticShape{2},
                                                    StaticShape{2}};
    OV_EXPECT_THROW(shape_inference(op.get(), static_input_shapes),
                    NodeValidationFailure,
                    HasSubstr("The sizes of `subsequence_begins` and `la_block_indices_begins` should be the same."));
}

TEST(StaticShapeInferenceTest, PagedSelectiveSSMProcessedTokensAndCacheIntervalMismatch) {
    const auto op = make_paged_selective_ssm();

    std::vector<StaticShape> static_input_shapes = {StaticShape{4},
                                                    StaticShape{6, 4},
                                                    StaticShape{6, 2, 16},
                                                    StaticShape{6, 4, 8},
                                                    StaticShape{6, 2, 16},
                                                    StaticShape{3, 4, 8, 16},
                                                    StaticShape{3},
                                                    StaticShape{3},
                                                    StaticShape{3},
                                                    StaticShape{2},
                                                    StaticShape{3}};
    OV_EXPECT_THROW(shape_inference(op.get(), static_input_shapes),
                    NodeValidationFailure,
                    HasSubstr("The sizes of `num_processed_tokens` and `cache_interval` should be the same."));
}

TEST(StaticShapeInferenceTest, PagedSelectiveSSMSubsequenceBeginsAndProcessedTokensMismatch) {
    const auto op = make_paged_selective_ssm();

    std::vector<StaticShape> static_input_shapes = {StaticShape{4},
                                                    StaticShape{6, 4},
                                                    StaticShape{6, 2, 16},
                                                    StaticShape{6, 4, 8},
                                                    StaticShape{6, 2, 16},
                                                    StaticShape{3, 4, 8, 16},
                                                    StaticShape{3},
                                                    StaticShape{3},
                                                    StaticShape{3},
                                                    StaticShape{3},
                                                    StaticShape{3}};
    OV_EXPECT_THROW(shape_inference(op.get(), static_input_shapes),
                    NodeValidationFailure,
                    HasSubstr("The size of `subsequence_begins` should be one larger than `num_processed_tokens`."));
}

TEST(StaticShapeInferenceTest, PagedSelectiveSSMMultipleGroups) {
    // 8 heads split across 4 groups (2 heads per group).
    const auto op = make_paged_selective_ssm();

    std::vector<StaticShape> static_input_shapes = {StaticShape{8},
                                                    StaticShape{5, 8},
                                                    StaticShape{5, 4, 16},
                                                    StaticShape{5, 8, 8},
                                                    StaticShape{5, 4, 16},
                                                    StaticShape{3, 8, 8, 16},
                                                    StaticShape{2},
                                                    StaticShape{4},
                                                    StaticShape{2},
                                                    StaticShape{1},
                                                    StaticShape{1}};
    const auto static_output_shapes = shape_inference(op.get(), static_input_shapes);
    ASSERT_EQ(static_output_shapes.size(), 1);
    EXPECT_EQ(static_output_shapes[0], StaticShape({5, 8, 8}));
}

TEST(StaticShapeInferenceTest, PagedSelectiveSSMMultiQueryStyleSingleGroup) {
    // All 8 heads share a single group (MQA-style).
    const auto op = make_paged_selective_ssm();

    std::vector<StaticShape> static_input_shapes = {StaticShape{8},
                                                    StaticShape{5, 8},
                                                    StaticShape{5, 1, 16},
                                                    StaticShape{5, 8, 8},
                                                    StaticShape{5, 1, 16},
                                                    StaticShape{3, 8, 8, 16},
                                                    StaticShape{2},
                                                    StaticShape{4},
                                                    StaticShape{2},
                                                    StaticShape{1},
                                                    StaticShape{1}};
    const auto static_output_shapes = shape_inference(op.get(), static_input_shapes);
    ASSERT_EQ(static_output_shapes.size(), 1);
    EXPECT_EQ(static_output_shapes[0], StaticShape({5, 8, 8}));
}

TEST(StaticShapeInferenceTest, PagedSelectiveSSMMultipleSequencesWithDifferentBlockCounts) {
    // A batch of 3 sequences, each addressing a different number of blocks in the shared state table.
    const auto op = make_paged_selective_ssm();

    std::vector<StaticShape> static_input_shapes = {StaticShape{4},
                                                    StaticShape{9, 4},
                                                    StaticShape{9, 2, 16},
                                                    StaticShape{9, 4, 8},
                                                    StaticShape{9, 2, 16},
                                                    StaticShape{10, 4, 8, 16},
                                                    StaticShape{4},
                                                    StaticShape{9},
                                                    StaticShape{4},
                                                    StaticShape{3},
                                                    StaticShape{3}};
    const auto static_output_shapes = shape_inference(op.get(), static_input_shapes);
    ASSERT_EQ(static_output_shapes.size(), 1);
    EXPECT_EQ(static_output_shapes[0], StaticShape({9, 4, 8}));
}
