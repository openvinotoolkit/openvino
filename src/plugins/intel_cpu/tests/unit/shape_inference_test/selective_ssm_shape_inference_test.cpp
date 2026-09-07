// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "selective_ssm_shape_inference.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/op/parameter.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using ov::op::v0::Parameter;
using testing::HasSubstr;

namespace {

std::shared_ptr<op::internal::SelectiveSSM> make_selective_ssm() {
    const auto dynamic = PartialShape::dynamic();
    auto A = std::make_shared<Parameter>(element::f32, dynamic);
    auto dt = std::make_shared<Parameter>(element::f32, dynamic);
    auto B = std::make_shared<Parameter>(element::f32, dynamic);
    auto x = std::make_shared<Parameter>(element::f32, dynamic);
    auto C = std::make_shared<Parameter>(element::f32, dynamic);
    auto state = std::make_shared<Parameter>(element::f32, dynamic);
    return std::make_shared<op::internal::SelectiveSSM>(A, dt, B, x, C, state);
}

}  // namespace

TEST(StaticShapeInferenceTest, SelectiveSSMBasic) {
    const auto op = make_selective_ssm();

    std::vector<StaticShape> static_input_shapes = {StaticShape{4},
                                                    StaticShape{6, 5, 4},
                                                    StaticShape{6, 5, 2, 16},
                                                    StaticShape{6, 5, 4, 8},
                                                    StaticShape{6, 5, 2, 16},
                                                    StaticShape{6, 4, 8, 16}};
    const auto static_output_shapes = shape_inference(op.get(), static_input_shapes);
    ASSERT_EQ(static_output_shapes.size(), 2);
    EXPECT_EQ(static_output_shapes[0], StaticShape({6, 5, 4, 8}));
    EXPECT_EQ(static_output_shapes[1], StaticShape({6, 4, 8, 16}));
}

TEST(StaticShapeInferenceTest, SelectiveSSMInvalidARank) {
    const auto op = make_selective_ssm();

    std::vector<StaticShape> static_input_shapes = {StaticShape{4, 1},
                                                    StaticShape{6, 5, 4},
                                                    StaticShape{6, 5, 2, 16},
                                                    StaticShape{6, 5, 4, 8},
                                                    StaticShape{6, 5, 2, 16},
                                                    StaticShape{6, 4, 8, 16}};
    EXPECT_THROW(shape_inference(op.get(), static_input_shapes), NodeValidationFailure);
}

TEST(StaticShapeInferenceTest, SelectiveSSMInvalidDtRank) {
    const auto op = make_selective_ssm();

    std::vector<StaticShape> static_input_shapes = {StaticShape{4},
                                                    StaticShape{6, 4},
                                                    StaticShape{6, 5, 2, 16},
                                                    StaticShape{6, 5, 4, 8},
                                                    StaticShape{6, 5, 2, 16},
                                                    StaticShape{6, 4, 8, 16}};
    EXPECT_THROW(shape_inference(op.get(), static_input_shapes), NodeValidationFailure);
}

TEST(StaticShapeInferenceTest, SelectiveSSMInvalidBRank) {
    const auto op = make_selective_ssm();

    std::vector<StaticShape> static_input_shapes = {StaticShape{4},
                                                    StaticShape{6, 5, 4},
                                                    StaticShape{6, 5, 2},
                                                    StaticShape{6, 5, 4, 8},
                                                    StaticShape{6, 5, 2, 16},
                                                    StaticShape{6, 4, 8, 16}};
    EXPECT_THROW(shape_inference(op.get(), static_input_shapes), NodeValidationFailure);
}

TEST(StaticShapeInferenceTest, SelectiveSSMInvalidXRank) {
    const auto op = make_selective_ssm();

    std::vector<StaticShape> static_input_shapes = {StaticShape{4},
                                                    StaticShape{6, 5, 4},
                                                    StaticShape{6, 5, 2, 16},
                                                    StaticShape{6, 5, 4},
                                                    StaticShape{6, 5, 2, 16},
                                                    StaticShape{6, 4, 8, 16}};
    EXPECT_THROW(shape_inference(op.get(), static_input_shapes), NodeValidationFailure);
}

TEST(StaticShapeInferenceTest, SelectiveSSMInvalidCRank) {
    const auto op = make_selective_ssm();

    std::vector<StaticShape> static_input_shapes = {StaticShape{4},
                                                    StaticShape{6, 5, 4},
                                                    StaticShape{6, 5, 2, 16},
                                                    StaticShape{6, 5, 4, 8},
                                                    StaticShape{6, 5, 2},
                                                    StaticShape{6, 4, 8, 16}};
    EXPECT_THROW(shape_inference(op.get(), static_input_shapes), NodeValidationFailure);
}

TEST(StaticShapeInferenceTest, SelectiveSSMInvalidRecurrentStateRank) {
    const auto op = make_selective_ssm();

    std::vector<StaticShape> static_input_shapes = {StaticShape{4},
                                                    StaticShape{6, 5, 4},
                                                    StaticShape{6, 5, 2, 16},
                                                    StaticShape{6, 5, 4, 8},
                                                    StaticShape{6, 5, 2, 16},
                                                    StaticShape{6, 4, 8}};
    EXPECT_THROW(shape_inference(op.get(), static_input_shapes), NodeValidationFailure);
}

TEST(StaticShapeInferenceTest, SelectiveSSMBatchMismatch) {
    const auto op = make_selective_ssm();

    std::vector<StaticShape> static_input_shapes = {StaticShape{4},
                                                    StaticShape{7, 5, 4},
                                                    StaticShape{6, 5, 2, 16},
                                                    StaticShape{6, 5, 4, 8},
                                                    StaticShape{6, 5, 2, 16},
                                                    StaticShape{6, 4, 8, 16}};
    OV_EXPECT_THROW(shape_inference(op.get(), static_input_shapes),
                    NodeValidationFailure,
                    HasSubstr("The batch dimension of `dt`, `B`, `x`, `C` and `recurrent_state` should be the same."));
}

TEST(StaticShapeInferenceTest, SelectiveSSMSeqLenMismatch) {
    const auto op = make_selective_ssm();

    std::vector<StaticShape> static_input_shapes = {StaticShape{4},
                                                    StaticShape{6, 7, 4},
                                                    StaticShape{6, 5, 2, 16},
                                                    StaticShape{6, 5, 4, 8},
                                                    StaticShape{6, 5, 2, 16},
                                                    StaticShape{6, 4, 8, 16}};
    OV_EXPECT_THROW(shape_inference(op.get(), static_input_shapes),
                    NodeValidationFailure,
                    HasSubstr("The sequence length of `dt`, `B`, `x` and `C` should be the same."));
}

TEST(StaticShapeInferenceTest, SelectiveSSMNumHeadsMismatch) {
    const auto op = make_selective_ssm();

    std::vector<StaticShape> static_input_shapes = {StaticShape{5},
                                                    StaticShape{6, 5, 4},
                                                    StaticShape{6, 5, 2, 16},
                                                    StaticShape{6, 5, 4, 8},
                                                    StaticShape{6, 5, 2, 16},
                                                    StaticShape{6, 4, 8, 16}};
    OV_EXPECT_THROW(shape_inference(op.get(), static_input_shapes),
                    NodeValidationFailure,
                    HasSubstr("The number of heads of `A`, `dt`, `x` and `recurrent_state` should be the same."));
}

TEST(StaticShapeInferenceTest, SelectiveSSMHeadDimMismatch) {
    const auto op = make_selective_ssm();

    std::vector<StaticShape> static_input_shapes = {StaticShape{4},
                                                    StaticShape{6, 5, 4},
                                                    StaticShape{6, 5, 2, 16},
                                                    StaticShape{6, 5, 4, 8},
                                                    StaticShape{6, 5, 2, 16},
                                                    StaticShape{6, 4, 10, 16}};
    OV_EXPECT_THROW(shape_inference(op.get(), static_input_shapes),
                    NodeValidationFailure,
                    HasSubstr("The head dimension of `x` and `recurrent_state` should be the same."));
}

TEST(StaticShapeInferenceTest, SelectiveSSMNumGroupsMismatch) {
    const auto op = make_selective_ssm();

    std::vector<StaticShape> static_input_shapes = {StaticShape{4},
                                                    StaticShape{6, 5, 4},
                                                    StaticShape{6, 5, 2, 16},
                                                    StaticShape{6, 5, 4, 8},
                                                    StaticShape{6, 5, 3, 16},
                                                    StaticShape{6, 4, 8, 16}};
    OV_EXPECT_THROW(shape_inference(op.get(), static_input_shapes),
                    NodeValidationFailure,
                    HasSubstr("The number of groups of `B` and `C` should be the same."));
}

TEST(StaticShapeInferenceTest, SelectiveSSMZeroGroups) {
    const auto op = make_selective_ssm();

    std::vector<StaticShape> static_input_shapes = {StaticShape{4},
                                                    StaticShape{6, 5, 4},
                                                    StaticShape{6, 5, 0, 16},
                                                    StaticShape{6, 5, 4, 8},
                                                    StaticShape{6, 5, 0, 16},
                                                    StaticShape{6, 4, 8, 16}};
    OV_EXPECT_THROW(shape_inference(op.get(), static_input_shapes),
                    NodeValidationFailure,
                    HasSubstr("The number of groups must be greater than zero."));
}

TEST(StaticShapeInferenceTest, SelectiveSSMStateSizeMismatch) {
    const auto op = make_selective_ssm();

    std::vector<StaticShape> static_input_shapes = {StaticShape{4},
                                                    StaticShape{6, 5, 4},
                                                    StaticShape{6, 5, 2, 16},
                                                    StaticShape{6, 5, 4, 8},
                                                    StaticShape{6, 5, 2, 32},
                                                    StaticShape{6, 4, 8, 16}};
    OV_EXPECT_THROW(shape_inference(op.get(), static_input_shapes),
                    NodeValidationFailure,
                    HasSubstr("The state size of `B`, `C` and `recurrent_state` should be the same."));
}

TEST(StaticShapeInferenceTest, SelectiveSSMHeadsNotDivisibleByGroups) {
    const auto op = make_selective_ssm();

    std::vector<StaticShape> static_input_shapes = {StaticShape{4},
                                                    StaticShape{6, 5, 4},
                                                    StaticShape{6, 5, 3, 16},
                                                    StaticShape{6, 5, 4, 8},
                                                    StaticShape{6, 5, 3, 16},
                                                    StaticShape{6, 4, 8, 16}};
    OV_EXPECT_THROW(shape_inference(op.get(), static_input_shapes),
                    NodeValidationFailure,
                    HasSubstr("The number of heads should be divisible by the number of groups."));
}

TEST(StaticShapeInferenceTest, SelectiveSSMMultipleGroups) {
    // 8 heads split across 4 groups (2 heads per group).
    const auto op = make_selective_ssm();

    std::vector<StaticShape> static_input_shapes = {StaticShape{8},
                                                    StaticShape{2, 3, 8},
                                                    StaticShape{2, 3, 4, 16},
                                                    StaticShape{2, 3, 8, 8},
                                                    StaticShape{2, 3, 4, 16},
                                                    StaticShape{2, 8, 8, 16}};
    const auto static_output_shapes = shape_inference(op.get(), static_input_shapes);
    ASSERT_EQ(static_output_shapes.size(), 2);
    EXPECT_EQ(static_output_shapes[0], StaticShape({2, 3, 8, 8}));
    EXPECT_EQ(static_output_shapes[1], StaticShape({2, 8, 8, 16}));
}

TEST(StaticShapeInferenceTest, SelectiveSSMMultiQueryStyleSingleGroup) {
    // All 8 heads share a single group (MQA-style).
    const auto op = make_selective_ssm();

    std::vector<StaticShape> static_input_shapes = {StaticShape{8},
                                                    StaticShape{2, 4, 8},
                                                    StaticShape{2, 4, 1, 16},
                                                    StaticShape{2, 4, 8, 8},
                                                    StaticShape{2, 4, 1, 16},
                                                    StaticShape{2, 8, 8, 16}};
    const auto static_output_shapes = shape_inference(op.get(), static_input_shapes);
    ASSERT_EQ(static_output_shapes.size(), 2);
    EXPECT_EQ(static_output_shapes[0], StaticShape({2, 4, 8, 8}));
    EXPECT_EQ(static_output_shapes[1], StaticShape({2, 8, 8, 16}));
}

TEST(StaticShapeInferenceTest, SelectiveSSMSingleHeadSingleGroup) {
    const auto op = make_selective_ssm();

    std::vector<StaticShape> static_input_shapes = {StaticShape{1},
                                                    StaticShape{3, 2, 1},
                                                    StaticShape{3, 2, 1, 8},
                                                    StaticShape{3, 2, 1, 4},
                                                    StaticShape{3, 2, 1, 8},
                                                    StaticShape{3, 1, 4, 8}};
    const auto static_output_shapes = shape_inference(op.get(), static_input_shapes);
    ASSERT_EQ(static_output_shapes.size(), 2);
    EXPECT_EQ(static_output_shapes[0], StaticShape({3, 2, 1, 4}));
    EXPECT_EQ(static_output_shapes[1], StaticShape({3, 1, 4, 8}));
}

TEST(StaticShapeInferenceTest, SelectiveSSMDecodeStepSingleToken) {
    // Single-step decode: sequence length is 1.
    const auto op = make_selective_ssm();

    std::vector<StaticShape> static_input_shapes = {StaticShape{4},
                                                    StaticShape{2, 1, 4},
                                                    StaticShape{2, 1, 2, 16},
                                                    StaticShape{2, 1, 4, 8},
                                                    StaticShape{2, 1, 2, 16},
                                                    StaticShape{2, 4, 8, 16}};
    const auto static_output_shapes = shape_inference(op.get(), static_input_shapes);
    ASSERT_EQ(static_output_shapes.size(), 2);
    EXPECT_EQ(static_output_shapes[0], StaticShape({2, 1, 4, 8}));
    EXPECT_EQ(static_output_shapes[1], StaticShape({2, 4, 8, 16}));
}
