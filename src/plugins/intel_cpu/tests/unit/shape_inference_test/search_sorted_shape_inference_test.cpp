// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using ov::op::v0::Constant;
using ov::op::v0::Parameter;
using testing::HasSubstr;

class SearchSortedShapeInferenceTest : public OpStaticShapeInferenceTest<op::v15::SearchSorted> {};

TEST_F(SearchSortedShapeInferenceTest, same_dimensions_nd_inputs) {
    const auto sorted = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto values = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto op = make_op(sorted, values);
    const auto input_shapes = StaticShapeVector{StaticShape{1, 3, 6}, StaticShape{1, 3, 6}};
    const auto output_shapes = shape_inference(op.get(), input_shapes);
    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({1, 3, 6}));
}

TEST_F(SearchSortedShapeInferenceTest, scalar_values) {
    const auto sorted = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto values = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto op = make_op(sorted, values);
    const auto input_shapes = StaticShapeVector{StaticShape{3}, StaticShape{}};
    const auto output_shapes = shape_inference(op.get(), input_shapes);
    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape{});
}

TEST_F(SearchSortedShapeInferenceTest, different_last_dim) {
    const auto sorted = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto values = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto op = make_op(sorted, values);
    const auto input_shapes = StaticShapeVector{StaticShape{1, 3, 7, 100}, StaticShape{1, 3, 7, 10}};
    const auto output_shapes = shape_inference(op.get(), input_shapes);
    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({1, 3, 7, 10}));
}

TEST_F(SearchSortedShapeInferenceTest, 1d_inputs) {
    const auto sorted = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto values = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto op = make_op(sorted, values);
    const auto input_shapes = StaticShapeVector{StaticShape{5}, StaticShape{20}};
    const auto output_shapes = shape_inference(op.get(), input_shapes);
    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({20}));
}

TEST_F(SearchSortedShapeInferenceTest, 1d_sequence) {
    const auto sorted = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto values = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto op = make_op(sorted, values);
    const auto input_shapes = StaticShapeVector{StaticShape{50}, StaticShape{1, 3, 7, 10}};
    const auto output_shapes = shape_inference(op.get(), input_shapes);
    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({1, 3, 7, 10}));
}

TEST_F(SearchSortedShapeInferenceTest, element_type_consistency_validation) {
    const auto sorted = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto values = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    OV_EXPECT_THROW(std::ignore = make_op(sorted, values),
                    NodeValidationFailure,
                    testing::HasSubstr("must have the same element type"));
}

TEST_F(SearchSortedShapeInferenceTest, input_shapes_ranks_validation) {
    const auto sorted = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto values = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto op = make_op(sorted, values);
    const auto input_shapes = StaticShapeVector{StaticShape{1, 3, 6}, StaticShape{1, 3, 6, 7}};
    OV_EXPECT_THROW(std::ignore = shape_inference(op.get(), input_shapes),
                    NodeValidationFailure,
                    testing::HasSubstr("the ranks of the inputs have to be compatible"));
}

TEST_F(SearchSortedShapeInferenceTest, input_shapes_compatibility) {
    const auto sorted = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto values = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto op = make_op(sorted, values);
    const auto input_shapes = StaticShapeVector{StaticShape{1, 3, 6}, StaticShape{1, 6, 6}};
    OV_EXPECT_THROW(std::ignore = shape_inference(op.get(), input_shapes),
                    NodeValidationFailure,
                    testing::HasSubstr("All dimensions but the last one have to be compatible"));
}

TEST_F(SearchSortedShapeInferenceTest, scalar_sorted_sequence) {
    const auto sorted = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto values = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto op = make_op(sorted, values);
    const auto input_shapes = StaticShapeVector{StaticShape{}, StaticShape{1, 6, 6}};
    OV_EXPECT_THROW(std::ignore = shape_inference(op.get(), input_shapes),
                    NodeValidationFailure,
                    testing::HasSubstr("The sorted sequence input cannot be a scalar"));
}

TEST_F(SearchSortedShapeInferenceTest, scalar_values_and_ND_sequence) {
    const auto sorted = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto values = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto op = make_op(sorted, values);
    const auto input_shapes = StaticShapeVector{StaticShape{2, 3}, StaticShape{}};
    OV_EXPECT_THROW(std::ignore = shape_inference(op.get(), input_shapes),
                    NodeValidationFailure,
                    testing::HasSubstr("the ranks of the inputs have to be compatible"));
}
