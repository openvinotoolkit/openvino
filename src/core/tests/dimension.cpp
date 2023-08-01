// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/dimension.hpp"

#include "gtest/gtest.h"
#include "openvino/core/dimension_tracker.hpp"

using namespace std;
using namespace ov;

TEST(dimension, broadcast_merge_static_1_and_10) {
    Dimension result;
    Dimension one(1), ten(10);
    bool success = Dimension::broadcast_merge(result, one, ten);

    EXPECT_TRUE(success);
    EXPECT_EQ(result, ten);
}

TEST(dimension, broadcast_merge_static_1_5_and_10_15) {
    Dimension result;
    Dimension one(1, 5), ten(10, 15);
    bool success = Dimension::broadcast_merge(result, one, ten);

    EXPECT_TRUE(success);
    EXPECT_EQ(result, ten);
}

TEST(dimension, broadcast_merge_static_1_12_and_10_15) {
    Dimension result;
    Dimension one(1, 12), ten(10, 15);
    bool success = Dimension::broadcast_merge(result, one, ten);

    EXPECT_TRUE(success);
    EXPECT_EQ(result, ten);
}

TEST(dimension, broadcast_merge_static_7_12_and_10_15) {
    Dimension result;
    Dimension one(7, 12), ten(10, 15);
    bool success = Dimension::broadcast_merge(result, one, ten);

    EXPECT_TRUE(success);
    EXPECT_EQ(result, Dimension(10, 12));
}

TEST(dimension, broadcast_merge_static_0_12_and_1_15) {
    Dimension result;
    Dimension one(0, 12), ten(1, 15);
    bool success = Dimension::broadcast_merge(result, one, ten);

    EXPECT_TRUE(success);
    EXPECT_EQ(result, Dimension(0, 15));
}

TEST(dimension, dimension_mul_operator_ordinary_intervals) {
    Dimension interval_1(0, 10);
    Dimension interval_2(2, 100);
    Dimension ref_value(0, 1000);
    EXPECT_EQ(ref_value, interval_1 * interval_2);
}

TEST(dimension, dimension_mul_operator_1) {
    Dimension fully_dynamic_dim(-1);
    Dimension two(2);
    Dimension ref_value(-1);
    EXPECT_EQ(ref_value, fully_dynamic_dim * two);
}

TEST(dimension, dimension_mul_operator_2) {
    // overflow happens and clip_times keeps result within in64 limits
    // (Interval::s_max - 1) * 2 = 9223372036854775806 * 2 = 18446744073709551612
    // arithmetical result does not fit into int64, is clipped into int64_max
    Dimension large_interval(2, Interval::s_max - 1);
    Dimension two(2);
    Dimension ref_value(4, Interval::s_max);
    EXPECT_EQ(ref_value, large_interval * two);
}

TEST(dimension, dimension_mul_operator_3) {
    // no overflow
    // (int64_max / 2) * 2= 4611686018427387903 * 2 = 9223372036854775806 = int64_max - 1
    Dimension large_interval(2, ov::Interval::s_max / 2);
    Dimension two(2);
    Dimension ref_value(4, ov::Interval::s_max - 1);
    EXPECT_EQ(ref_value, large_interval * two);
}

TEST(dimension, dimension_mul_operator_4) {
    // overflow happens and clip_times keeps result within in64 limits
    // (int64_max / 2 + 1) * 2 = 4611686018427387904 * 2 = 9223372036854775808 = int64_max + 1
    // 9223372036854775808 does not fit into int64, is clipped into int64_max
    Dimension large_interval(2, ov::Interval::s_max / 2 + 1);
    Dimension two(2);
    Dimension ref_value(4, ov::Interval::s_max);
    EXPECT_EQ(ref_value, large_interval * two);
}

TEST(dimension, dimension_mul_operator_5) {
    // (int64_max / 3 + 2) = 3074457345618258604 * 3 = 9223372036854775812 = int64_max + 5
    // overflow happens and clip_times keeps result within in64 limits
    // 9223372036854775812 does not fit into int64, is clipped into int64_max
    Dimension large_interval(2, ov::Interval::s_max / 3 + 2);
    Dimension three(3);
    Dimension ref_value(6, ov::Interval::s_max);
    EXPECT_EQ(ref_value, large_interval * three);
}

TEST(dimension, division_of_static_dims_twenty_three_div_three_eq_seven) {
    Dimension twenty_three(23);
    Dimension::value_type three(3);
    Dimension empty(8, 7);
    EXPECT_EQ(empty, twenty_three / three);
}

TEST(dimension, division_of_static_dims) {
    Dimension seven(7);
    Dimension::value_type four(4);
    Dimension empty(2, 1);
    EXPECT_EQ(seven / four, empty);
}

TEST(dimension, dimension_equality) {
    auto te = make_shared<TableOfEquivalence>();
    DimensionTracker dt(te);

    // labeling dimensions
    Dimension A, B, C;
    dt.set_up_for_tracking(A);
    dt.set_up_for_tracking(B);
    dt.set_up_for_tracking(C);

    // checking labels are unique
    EXPECT_NE(DimensionTracker::get_label(A), no_label);
    EXPECT_NE(DimensionTracker::get_label(B), no_label);
    EXPECT_NE(DimensionTracker::get_label(C), no_label);
    EXPECT_NE(DimensionTracker::get_label(A), DimensionTracker::get_label(B));
    EXPECT_NE(DimensionTracker::get_label(B), DimensionTracker::get_label(C));
    EXPECT_NE(DimensionTracker::get_label(A), DimensionTracker::get_label(C));
    EXPECT_EQ(DimensionTracker::get_label(A), DimensionTracker::get_label(A));
    EXPECT_EQ(DimensionTracker::get_label(B), DimensionTracker::get_label(B));
    EXPECT_EQ(DimensionTracker::get_label(C), DimensionTracker::get_label(C));

    // setting A == B and B == C
    te->set_as_equal(A, B);
    te->set_as_equal(C, B);

    // expected to see A == B, B == C and A == C
    EXPECT_TRUE(te->are_equal(A, B));
    EXPECT_TRUE(te->are_equal(A, C));
    EXPECT_TRUE(te->are_equal(B, C));

    // clear up all the tracking info
    DimensionTracker::reset_tracking_info(A);
    DimensionTracker::reset_tracking_info(B);
    DimensionTracker::reset_tracking_info(C);

    // expected to have no label
    EXPECT_EQ(DimensionTracker::get_label(A), no_label);
    EXPECT_EQ(DimensionTracker::get_label(B), no_label);
    EXPECT_EQ(DimensionTracker::get_label(C), no_label);
}
