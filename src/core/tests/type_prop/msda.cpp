// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/msda.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/openvino.hpp"

namespace ov::test {
namespace {

std::shared_ptr<op::internal::MSDA> make_msda(const element::Type& et,
                                              const PartialShape& value,
                                              const PartialShape& spatial_shapes,
                                              const PartialShape& level_start_index,
                                              const PartialShape& locations,
                                              const PartialShape& weights,
                                              const element::Type& index_et = element::i32,
                                              const element::Type& locations_et = element::dynamic) {
    return std::make_shared<op::internal::MSDA>(
        std::make_shared<op::v0::Parameter>(et, value),
        std::make_shared<op::v0::Parameter>(index_et, spatial_shapes),
        std::make_shared<op::v0::Parameter>(index_et, level_start_index),
        std::make_shared<op::v0::Parameter>(locations_et == element::dynamic ? et : locations_et, locations),
        std::make_shared<op::v0::Parameter>(et, weights));
}

}  // namespace

TEST(type_prop, msda_static_f32) {
    const auto op = make_msda(element::f32,
                              Shape{2, 31, 4, 8},
                              Shape{2, 2},
                              Shape{2},
                              Shape{2, 6, 4, 2, 3, 2},
                              Shape{2, 6, 4, 2, 3});

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({2, 6, 32}));
}

TEST(type_prop, msda_static_f16_i64_indices) {
    const auto op = make_msda(element::f16,
                              Shape{1, 20, 2, 16},
                              Shape{1, 2},
                              Shape{1},
                              Shape{1, 9, 2, 1, 4, 2},
                              Shape{1, 9, 2, 1, 4},
                              element::i64);

    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({1, 9, 32}));
}

TEST(type_prop, msda_dims_resolved_from_other_inputs) {
    const auto op = make_msda(element::f32,
                              PartialShape{-1, -1, 4, 8},
                              PartialShape{-1, 2},
                              PartialShape{-1},
                              PartialShape{2, -1, -1, 2, 3, 2},
                              PartialShape{-1, 6, 4, -1, -1});

    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({2, 6, 32}));
}

TEST(type_prop, msda_interval_dims) {
    const auto op = make_msda(element::f32,
                              PartialShape{{1, 4}, -1, 4, 8},
                              PartialShape{2, 2},
                              PartialShape{2},
                              PartialShape{{2, 8}, {10, 20}, 4, 2, 3, 2},
                              PartialShape{-1, {15, 30}, 4, 2, 3});

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{{2, 4}, {15, 20}, 32}));
}

TEST(type_prop, msda_dynamic_rank_inputs) {
    const auto op = make_msda(element::f32,
                              PartialShape::dynamic(),
                              PartialShape::dynamic(),
                              PartialShape::dynamic(),
                              PartialShape::dynamic(),
                              PartialShape{1, 7, 2, 1, 4});

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1, 7, -1}));
}

TEST(type_prop, msda_invalid_value_rank) {
    OV_EXPECT_THROW(std::ignore = make_msda(element::f32,
                                            Shape{1, 20, 16},
                                            Shape{1, 2},
                                            Shape{1},
                                            Shape{1, 7, 2, 1, 4, 2},
                                            Shape{1, 7, 2, 1, 4}),
                    NodeValidationFailure,
                    testing::HasSubstr("MSDA value input must be 4D"));
}

TEST(type_prop, msda_invalid_spatial_shapes) {
    OV_EXPECT_THROW(std::ignore = make_msda(element::f32,
                                            Shape{1, 20, 2, 8},
                                            Shape{1, 3},
                                            Shape{1},
                                            Shape{1, 7, 2, 1, 4, 2},
                                            Shape{1, 7, 2, 1, 4}),
                    NodeValidationFailure,
                    testing::HasSubstr("value_spatial_shapes input must have shape (num_levels, 2)"));
}

TEST(type_prop, msda_invalid_data_type) {
    OV_EXPECT_THROW(std::ignore = make_msda(element::i32,
                                            Shape{1, 20, 2, 8},
                                            Shape{1, 2},
                                            Shape{1},
                                            Shape{1, 7, 2, 1, 4, 2},
                                            Shape{1, 7, 2, 1, 4}),
                    NodeValidationFailure,
                    testing::HasSubstr("MSDA value must have a floating point element type"));
}

TEST(type_prop, msda_mismatched_data_types) {
    OV_EXPECT_THROW(std::ignore = make_msda(element::f32,
                                            Shape{1, 20, 2, 8},
                                            Shape{1, 2},
                                            Shape{1},
                                            Shape{1, 7, 2, 1, 4, 2},
                                            Shape{1, 7, 2, 1, 4},
                                            element::i32,
                                            element::f16),
                    NodeValidationFailure,
                    testing::HasSubstr("must have the same element type"));
}

TEST(type_prop, msda_invalid_index_type) {
    OV_EXPECT_THROW(std::ignore = make_msda(element::f32,
                                            Shape{1, 20, 2, 8},
                                            Shape{1, 2},
                                            Shape{1},
                                            Shape{1, 7, 2, 1, 4, 2},
                                            Shape{1, 7, 2, 1, 4},
                                            element::f32),
                    NodeValidationFailure,
                    testing::HasSubstr("must have an integer element type"));
}

TEST(type_prop, msda_batch_mismatch) {
    OV_EXPECT_THROW(std::ignore = make_msda(element::f32,
                                            Shape{2, 20, 2, 8},
                                            Shape{1, 2},
                                            Shape{1},
                                            Shape{1, 7, 2, 1, 4, 2},
                                            Shape{1, 7, 2, 1, 4}),
                    NodeValidationFailure,
                    testing::HasSubstr("inconsistent batch size"));
}

TEST(type_prop, msda_heads_mismatch) {
    OV_EXPECT_THROW(std::ignore = make_msda(element::f32,
                                            Shape{1, 20, 4, 8},
                                            Shape{1, 2},
                                            Shape{1},
                                            Shape{1, 7, 2, 1, 4, 2},
                                            Shape{1, 7, 2, 1, 4}),
                    NodeValidationFailure,
                    testing::HasSubstr("inconsistent number of heads"));
}

TEST(type_prop, msda_levels_mismatch) {
    OV_EXPECT_THROW(std::ignore = make_msda(element::f32,
                                            Shape{1, 20, 2, 8},
                                            Shape{2, 2},
                                            Shape{2},
                                            Shape{1, 7, 2, 1, 4, 2},
                                            Shape{1, 7, 2, 1, 4}),
                    NodeValidationFailure,
                    testing::HasSubstr("inconsistent number of levels"));
}

TEST(type_prop, msda_queries_mismatch) {
    OV_EXPECT_THROW(std::ignore = make_msda(element::f32,
                                            Shape{1, 20, 2, 8},
                                            Shape{1, 2},
                                            Shape{1},
                                            Shape{1, 7, 2, 1, 4, 2},
                                            Shape{1, 5, 2, 1, 4}),
                    NodeValidationFailure,
                    testing::HasSubstr("inconsistent number of queries"));
}

TEST(type_prop, msda_points_mismatch) {
    OV_EXPECT_THROW(std::ignore = make_msda(element::f32,
                                            Shape{1, 20, 2, 8},
                                            Shape{1, 2},
                                            Shape{1},
                                            Shape{1, 7, 2, 1, 4, 2},
                                            Shape{1, 7, 2, 1, 3}),
                    NodeValidationFailure,
                    testing::HasSubstr("inconsistent number of points"));
}

TEST(type_prop, msda_invalid_locations_last_dim) {
    OV_EXPECT_THROW(std::ignore = make_msda(element::f32,
                                            Shape{1, 20, 2, 8},
                                            Shape{1, 2},
                                            Shape{1},
                                            Shape{1, 7, 2, 1, 4, 3},
                                            Shape{1, 7, 2, 1, 4}),
                    NodeValidationFailure,
                    testing::HasSubstr("MSDA sampling_locations input must have shape"));
}

TEST(type_prop, msda_invalid_weights_rank) {
    OV_EXPECT_THROW(std::ignore = make_msda(element::f32,
                                            Shape{1, 20, 2, 8},
                                            Shape{1, 2},
                                            Shape{1},
                                            Shape{1, 7, 2, 1, 4, 2},
                                            Shape{1, 7, 2, 4}),
                    NodeValidationFailure,
                    testing::HasSubstr("MSDA attention_weights input must be 5D"));
}

TEST(type_prop, msda_invalid_level_start_index_rank) {
    OV_EXPECT_THROW(std::ignore = make_msda(element::f32,
                                            Shape{1, 20, 2, 8},
                                            Shape{1, 2},
                                            Shape{1, 1},
                                            Shape{1, 7, 2, 1, 4, 2},
                                            Shape{1, 7, 2, 1, 4}),
                    NodeValidationFailure,
                    testing::HasSubstr("MSDA level_start_index input must be 1D"));
}

TEST(type_prop, msda_wrong_input_count) {
    const auto value = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 20, 2, 8});
    const auto spatial_shapes = std::make_shared<op::v0::Parameter>(element::i32, Shape{1, 2});
    const auto level_start_index = std::make_shared<op::v0::Parameter>(element::i32, Shape{1});
    const auto locations = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 7, 2, 1, 4, 2});
    const auto op = std::make_shared<op::internal::MSDA>();
    op->set_arguments(OutputVector{value, spatial_shapes, level_start_index, locations});
    OV_EXPECT_THROW(op->validate_and_infer_types(),
                    NodeValidationFailure,
                    testing::HasSubstr("MSDA must have 5 inputs"));
}

}  // namespace ov::test
