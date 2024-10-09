// Copyright (C) 2018-2024 Intel Corporation
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

class SearchSortedStaticTestSuite : public ::testing::TestWithParam<std::tuple<ov::Shape,       // sorted shape
                                                              ov::Shape,                        // values shape
                                                              ov::Shape>> {};                   // expected output shape

class SearchSortedStaticShapeInferenceTest: public OpStaticShapeInferenceTest<op::v15::SearchSorted> {};

TEST_P(SearchSortedStaticTestSuite, SearchSortedStaticShapeInference) {
    const auto& param = GetParam();
    const auto& sorted_shape = std::get<0>(param);
    const auto& values_shape = std::get<1>(param);
    const auto& expected_shape = std::get<2>(param);

    const auto sorted = std::make_shared<Parameter>(element::i64, sorted_shape);
    const auto values = std::make_shared<Parameter>(element::i64, values_shape);
    const auto op = std::make_shared<op::v15::SearchSorted>(sorted, values);
    const auto input_shapes = ShapeVector{sorted_shape, values_shape};
    auto shape_infer = make_shape_inference(op);
    const auto input_shape_refs = make_static_shape_refs(input_shapes);
    const auto output_shapes = *shape_infer->infer(input_shape_refs, make_tensor_accessor());

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape(expected_shape));
}

INSTANTIATE_TEST_SUITE_P(
    SearchSortedStaticShapeInferenceTests,
    SearchSortedStaticTestSuite,
    ::testing::Values(
        std::make_tuple(
            Shape{1, 3, 6},             // sorted shape
            Shape{1, 3, 6},             // values shape
            Shape{1, 3, 6}),            // expected shape
        std::make_tuple(
            Shape{1, 3, 7, 100},        // sorted shape
            Shape{1, 3, 7, 10},         // values shape
            Shape{1, 3, 7, 10}),        // expected shape
        std::make_tuple(
            Shape{5},                   // sorted shape
            Shape{20},                  // values shape
            Shape{20}),                 // expected shape
        std::make_tuple(
            Shape{50},                  // sorted shape
            Shape{1, 3, 7, 10},         // values shape
            Shape{1, 3, 7, 10})         // expected shape
    )
);

TEST(StaticShapeInferenceTest, SearchSorted_element_type_consistency_validation) {
    const auto sorted = std::make_shared<Parameter>(element::i64, ov::Shape{1, 3, 6});
    const auto values = std::make_shared<Parameter>(element::i32, ov::Shape{1, 3, 6});
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::v15::SearchSorted>(sorted, values),
                    NodeValidationFailure,
                    testing::HasSubstr("must have the same element type"));
}

TEST(StaticShapeInferenceTest, SearchSorted_input_shapes_ranks_validation) {
    const auto sorted_shape = ov::Shape{1, 3, 6};
    const auto values_shape = ov::Shape{1, 3, 6, 7};
    const auto sorted = std::make_shared<Parameter>(element::i32, sorted_shape);
    const auto values = std::make_shared<Parameter>(element::i32, values_shape);
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::v15::SearchSorted>(sorted, values),
                    NodeValidationFailure,
                    testing::HasSubstr("The inputs' ranks have to be compatible"));
}

TEST(StaticShapeInferenceTest, SearchSorted_input_shapes_compatibility) {
    const auto sorted_shape = ov::Shape{1, 3, 6};
    const auto values_shape = ov::Shape{1, 6, 6};
    const auto sorted = std::make_shared<Parameter>(element::i32, sorted_shape);
    const auto values = std::make_shared<Parameter>(element::i32, values_shape);
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::v15::SearchSorted>(sorted, values),
                    NodeValidationFailure,
                    testing::HasSubstr("All dimensions but the last one have to be compatible"));
}
