// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "openvino/core/interval.hpp"

using namespace std;
using namespace ov;

TEST(intervals, size) {
    EXPECT_TRUE(ov::Interval().size() > 0);
    EXPECT_TRUE(Interval(2).size() == 1);
    EXPECT_TRUE(Interval(1, 5).size() == 5);
    EXPECT_TRUE(Interval(3, 2).size() == 0);
    EXPECT_TRUE(Interval(3, 3).size() == 1);
}

TEST(intervals, contains) {
    Interval x(3, 10);
    for (auto i = x.get_min_val(); i <= x.get_max_val(); ++i) {
        EXPECT_TRUE(x.contains(i));
    }
    EXPECT_FALSE(x.contains(x.get_max_val() + 1));
    EXPECT_FALSE(x.contains(x.get_min_val() - 1));
    Interval empty(1, -1);
    EXPECT_TRUE(empty.empty());
    EXPECT_TRUE(Interval().contains(x));
}

TEST(intervals, equals) {
    EXPECT_TRUE(Interval(2, 5) == Interval(2, 5));
    EXPECT_FALSE(Interval(2, 5) != Interval(2, 5));
    EXPECT_FALSE(Interval(3) == Interval(5));
    EXPECT_TRUE(Interval(3) != Interval(5));
    Interval a(2);
    Interval b(a);
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a != b);
    Interval c(2, 4);
    b = c;
    EXPECT_TRUE(b == c);
    EXPECT_FALSE(b != c);
    EXPECT_EQ(Interval() & Interval(3), Interval(3));
    EXPECT_EQ(Interval() + Interval(5), Interval(5, Interval::s_max));
    EXPECT_TRUE(Interval(5, 3).empty());
    EXPECT_TRUE((Interval(3) & Interval(4)).empty());
    // All empty intervals are the same and stay empty
    EXPECT_EQ(Interval(5, 3), Interval(8, 1));
    EXPECT_EQ(Interval(5, 1) + Interval(2, 4), Interval(3, 1));
    EXPECT_EQ(Interval(5, 1) * Interval(2, 4), Interval(3, 1));
    EXPECT_EQ(Interval(5, 1) - Interval(2, 4), Interval(3, 1));
    EXPECT_EQ(Interval(5, 1) & Interval(2, 4), Interval(3, 1));
}

TEST(intervals, arithmetic) {
    Interval a(7, 10);
    Interval b(1, 5);
    Interval a_plus = a;
    auto a_plus_b = a + b;
    a_plus += b;
    EXPECT_TRUE(a_plus_b == a_plus);
    Interval::value_type min_plus = numeric_limits<Interval::value_type>::max();
    Interval::value_type max_plus = numeric_limits<Interval::value_type>::min();
    auto a_minus_b = a - b;
    Interval a_minus = a;
    a_minus -= b;
    EXPECT_TRUE(a_minus_b == a_minus);
    Interval::value_type min_minus = numeric_limits<Interval::value_type>::max();
    Interval::value_type max_minus = numeric_limits<Interval::value_type>::min();
    auto a_times_b = a * b;
    Interval a_times = a;
    a_times *= b;
    EXPECT_TRUE(a_times_b == a_times);
    Interval::value_type min_times = numeric_limits<Interval::value_type>::max();
    Interval::value_type max_times = numeric_limits<Interval::value_type>::min();
    // Manually collect sum, difference, and product min/max ranges and verify they correspond to
    // the computed intervals and that they are all members of the computed intervals.
    for (auto a_i = a.get_min_val(); a_i <= a.get_max_val(); ++a_i) {
        for (auto b_i = b.get_min_val(); b_i <= b.get_max_val(); ++b_i) {
            auto sum = a_i + b_i;
            EXPECT_TRUE(a_plus_b.contains(sum));
            if (sum < min_plus) {
                min_plus = sum;
            }
            if (sum > max_plus) {
                max_plus = sum;
            }
            auto minus = a_i - b_i;
            if (minus < 0) {
                EXPECT_FALSE(a_minus_b.contains(minus));
            } else {
                EXPECT_TRUE(a_minus_b.contains(minus));
            }
            if (minus < min_minus) {
                min_minus = minus;
            }
            if (minus > max_minus) {
                max_minus = minus;
            }
            min_minus = max(Interval::value_type(0), min_minus);

            auto times = a_i * b_i;
            EXPECT_TRUE(a_times_b.contains(times));
            if (times < min_times) {
                min_times = times;
            }
            if (times > max_times) {
                max_times = times;
            }
        }
    }
    EXPECT_TRUE(Interval(min_plus, max_plus) == a_plus_b);
    EXPECT_TRUE(Interval(min_minus, max_minus) == a_minus_b);
    EXPECT_TRUE(Interval(min_times, max_times) == a_times_b);
}

TEST(intervals, sets) {
    Interval a(1, 5);
    Interval b(3, 7);
    Interval a_int = a;
    auto a_int_b = a & b;
    a_int &= b;
    EXPECT_TRUE(a_int_b == a_int);
    Interval::value_type min_int = numeric_limits<Interval::value_type>::max();
    Interval::value_type max_int = numeric_limits<Interval::value_type>::min();
    // Manually collect the min/max of the intersection and make sure this corresponds to the
    // computed intersection
    for (auto a_i = a.get_min_val(); a_i <= a.get_max_val(); ++a_i) {
        for (auto b_i = b.get_min_val(); b_i <= b.get_max_val(); ++b_i) {
            if (a_i == b_i) {
                if (a_i < min_int) {
                    min_int = a_i;
                }
                if (a_i > max_int) {
                    max_int = a_i;
                }
                EXPECT_TRUE(a_int_b.contains(a_i));
            }
        }
    }
    EXPECT_TRUE(Interval(min_int, max_int) == a_int_b);
}

TEST(intervals, corner_cases) {
    Interval::value_type max = numeric_limits<Interval::value_type>::max();
    Interval almost_max(0, max - 10);
    Interval dynamic(0, max);
    Interval zero(0, 0);

    EXPECT_TRUE(almost_max + almost_max == dynamic);
    EXPECT_TRUE(dynamic + almost_max == dynamic);
    EXPECT_TRUE(almost_max + dynamic == dynamic);
    EXPECT_TRUE(dynamic - almost_max == dynamic);

    EXPECT_TRUE(dynamic * almost_max == dynamic);
    EXPECT_TRUE(almost_max * dynamic == dynamic);
    EXPECT_TRUE(zero * almost_max == zero);
    EXPECT_TRUE(almost_max * zero == zero);
}
