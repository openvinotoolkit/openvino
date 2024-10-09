// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/search_sorted.hpp"

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"

using namespace std;
using namespace ov;

#define EXPECT_THROW_SUBSTRING(SORTED, VALUES, SUBSTRING)                                           \
    OV_EXPECT_THROW_HAS_SUBSTRING(std::ignore = make_shared<op::v15::SearchSorted>(SORTED, VALUES), \
                                  NodeValidationFailure,                                            \
                                  SUBSTRING);

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

TEST(type_prop, search_sorted_shape_infer_sorted_1d) {
    PerformShapeTest({5}, {2, 3}, {2, 3});
}

TEST(type_prop, search_sorted_shape_infer_sorted_and_values_1d) {
    PerformShapeTest({5}, {20}, {20});
}

TEST(type_prop, search_sorted_shape_infer_sorted_1d_values_dynamic) {
    PerformShapeTest({8}, {-1, -1, 3}, {-1, -1, 3});
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

TEST(type_prop, search_sorted_shape_infer_both_dynamic_5) {
    PerformShapeTest({-1}, {-1, -1, 3}, {-1, -1, 3});
}

TEST(type_prop, search_sorted_shape_infer_different_types) {
    auto sorted = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 6});
    auto values = make_shared<ov::op::v0::Parameter>(element::i32, Shape{1, 3, 6});
    EXPECT_THROW_SUBSTRING(values, sorted, std::string("must have the same element type"));
}

TEST(type_prop, search_sorted_shape_infer_wrong_rank) {
    auto sorted = make_shared<ov::op::v0::Parameter>(element::i32, Shape{1, 1, 3, 6});
    auto values = make_shared<ov::op::v0::Parameter>(element::i32, Shape{1, 3, 6});
    EXPECT_THROW_SUBSTRING(sorted, values, std::string("The inputs' ranks have to be compatible"));
}

TEST(type_prop, search_sorted_shape_infer_wrong_dim) {
    auto sorted = make_shared<ov::op::v0::Parameter>(element::i32, Shape{1, 1, 3, 6});
    auto values = make_shared<ov::op::v0::Parameter>(element::i32, Shape{1, 1, 5, 6});
    EXPECT_THROW_SUBSTRING(sorted, values, std::string("All dimensions but the last one have to be compatible"));
}

#undef EXPECT_THROW_SUBSTRING