// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/search_sorted.hpp"

#include "common_test_utils/type_prop.hpp"

using namespace std;
using namespace ov;

#define EXPECT_THROW_SUBSTRING(STATEMENT, SUBSTRING)              \
    try {                                                         \
        STATEMENT;                                                \
        FAIL() << "Exception not thrown";                         \
    } catch (const NodeValidationFailure& error) {                \
        EXPECT_THAT(error.what(), testing::HasSubstr(SUBSTRING)); \
    } catch (...) {                                               \
        FAIL() << "Unexpected exception thrown";                  \
    }

static void PerformShapeTest(const PartialShape& sorted_shape,
                             const PartialShape& values_shape,
                             const PartialShape& expected_output_shape) {
    auto sorted = make_shared<op::v0::Parameter>(element::i32, sorted_shape);
    auto values = make_shared<op::v0::Parameter>(element::i32, values_shape);
    auto search_sorted_op = make_shared<op::v15::SearchSorted>(sorted, values);
    EXPECT_EQ(search_sorted_op->get_element_type(), element::i64);
    EXPECT_EQ(search_sorted_op->get_output_partial_shape(0), expected_output_shape);
}

TEST(type_prop, search_sorted_shape_infer_equal_inputs) {
    PerformShapeTest({1, 3, 6}, {1, 3, 6}, {1, 3, 6});
}

TEST(type_prop, search_sorted_shape_infer_sorted_dynamic) {
    PerformShapeTest(PartialShape::dynamic(), {1, 3, 6}, {1, 3, 6});
}

TEST(type_prop, search_sorted_shape_infer_values_dynamic) {
    PerformShapeTest({1, 3, 7, 5}, PartialShape::dynamic(), {1, 3, 7, -1});
}

TEST(type_prop, search_sorted_shape_infer_different_last_dim) {
    PerformShapeTest({1, 3, 7, 100}, {1, 3, 7, 10}, {1, 3, 7, 10});
}

TEST(type_prop, search_sorted_shape_infer_both_dynamic_1) {
    PerformShapeTest({1, -1, 7, -1}, {-1, 3, -1, 10}, {1, 3, 7, 10});
}

TEST(type_prop, search_sorted_shape_infer_both_dynamic_2) {
    PerformShapeTest({1, -1, 7, 50}, {-1, 3, -1, -1}, {1, 3, 7, -1});
}

TEST(type_prop, search_sorted_shape_infer_both_dynamic_3) {
    PerformShapeTest(PartialShape::dynamic(), PartialShape::dynamic(), PartialShape::dynamic());
}

TEST(type_prop, search_sorted_shape_infer_both_dynamic_4) {
    PerformShapeTest({-1, -1, 50}, {-1, -1, 20}, {-1, -1, 20});
}

TEST(type_prop, search_sorted_shape_infer_different_types) {
    auto sorted = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 6});
    auto values = make_shared<ov::op::v0::Parameter>(element::i32, Shape{1, 3, 6});
    EXPECT_THROW_SUBSTRING(make_shared<op::v15::SearchSorted>(values, sorted),
                           std::string("must have the same element type"));
}

TEST(type_prop, search_sorted_shape_infer_wrong_rank) {
    auto sorted = make_shared<ov::op::v0::Parameter>(element::i32, Shape{1, 1, 3, 6});
    auto values = make_shared<ov::op::v0::Parameter>(element::i32, Shape{1, 3, 6});
    EXPECT_THROW_SUBSTRING(make_shared<op::v15::SearchSorted>(sorted, values),
                           std::string("Sorted sequence and values have different ranks"));
}

TEST(type_prop, search_sorted_shape_infer_wrong_dim) {
    auto sorted = make_shared<ov::op::v0::Parameter>(element::i32, Shape{1, 1, 3, 6});
    auto values = make_shared<ov::op::v0::Parameter>(element::i32, Shape{1, 1, 5, 6});
    EXPECT_THROW_SUBSTRING(make_shared<op::v15::SearchSorted>(sorted, values), std::string(" different 2 dimension."));
}

#undef EXPECT_THROW_SUBSTRING