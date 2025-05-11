// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/utils/coordinate_range.hpp"

#include <algorithm>
#include <numeric>
#include <utility>

#include "gtest/gtest.h"
#include "openvino/core/coordinate.hpp"

using namespace ov;
using Index = size_t;
using ExpectedOutput = std::vector<std::pair<Index, ov::Coordinate>>;

///
///
///     SliceRange
///
///

TEST(coordinate_range, slice_range_shape0d) {
    const Shape s;
    const Coordinate start_corner(s.size());

    auto slice_range = ov::coordinates::slice(s, start_corner, s);
    auto it = slice_range.begin();
    EXPECT_EQ(it, begin(slice_range));
    EXPECT_FALSE(it == slice_range.end());
    auto v = *it;  // if it is not end it has to be dereferencable;
    (void)v;
    EXPECT_TRUE(++it == slice_range.end());
}

TEST(coordinate_range, slice_range_shape1d) {
    const Shape s{3};
    const Coordinate start_corner(s.size());

    const ExpectedOutput expected{{0, {0}}, {1, {1}}, {2, {2}}};
    ASSERT_EQ(expected.size(), shape_size(s)) << "check epxected data";

    auto expected_val = begin(expected);
    for (auto slice_range : ov::coordinates::slice(s, start_corner, s)) {
        auto index = slice_range.begin_index;
        for (size_t i = 0; i < slice_range.element_number; index += slice_range.step, ++i) {
            EXPECT_EQ(index, expected_val->first);
            ++expected_val;
        }
    }
    EXPECT_TRUE(expected_val == end(expected))
        << "not all expected values return, (" << std::distance(expected_val, end(expected)) << " is missing)";
}

TEST(coordinate_range, slice_range_shape2d) {
    const Shape s{2, 3};
    const Coordinate start_corner(s.size());

    // clang-format off
    const ExpectedOutput expected{
        {0, {0, 0}}, {1, {0, 1}}, {2, {0, 2}},
        {3, {1, 0}}, {4, {1, 1}}, {5, {1, 2}}};
    // clang-format on
    ASSERT_EQ(expected.size(), shape_size(s)) << "check epxected data";

    auto expected_val = begin(expected);
    for (auto slice_range : ov::coordinates::slice(s, start_corner, s)) {
        auto index = slice_range.begin_index;
        for (size_t i = 0; i < slice_range.element_number; index += slice_range.step, ++i) {
            EXPECT_EQ(index, expected_val->first);
            ++expected_val;
        }
    }
    EXPECT_TRUE(expected_val == end(expected))
        << "not all expected values return, (" << std::distance(expected_val, end(expected)) << " is missing)";
}

TEST(coordinate_range, slice_range_shape3d) {
    const Shape s{2, 3, 4};
    const Coordinate start_corner(s.size());

    // clang-format off
    const ExpectedOutput expected{
        {0, {0, 0, 0}},  {1, {0, 0, 1}},  {2, {0, 0, 2}},  {3, {0, 0, 3}},
        {4, {0, 1, 0}},  {5, {0, 1, 1}},  {6, {0, 1, 2}},  {7, {0, 1, 3}},
        {8, {0, 2, 0}},  {9, {0, 2, 1}},  {10, {0, 2, 2}}, {11, {0, 2, 3}},
        {12, {1, 0, 0}}, {13, {1, 0, 1}}, {14, {1, 0, 2}}, {15, {1, 0, 3}},
        {16, {1, 1, 0}}, {17, {1, 1, 1}}, {18, {1, 1, 2}}, {19, {1, 1, 3}},
        {20, {1, 2, 0}}, {21, {1, 2, 1}}, {22, {1, 2, 2}}, {23, {1, 2, 3}}};
    // clang-format on
    ASSERT_EQ(expected.size(), shape_size(s)) << "check epxected data";

    auto expected_val = begin(expected);
    for (auto slice_range : ov::coordinates::slice(s, start_corner, s)) {
        auto index = slice_range.begin_index;
        for (size_t i = 0; i < slice_range.element_number; index += slice_range.step, ++i) {
            EXPECT_EQ(index, expected_val->first);
            ++expected_val;
        }
    }
    EXPECT_TRUE(expected_val == end(expected));
}

TEST(coordinate_range, slice_range_zero_sized_axis) {
    const Shape s{2, 0, 4};
    const Coordinate start_corner(s.size());

    auto slice_range = ov::coordinates::slice(s, start_corner, s);
    auto it = slice_range.begin();
    EXPECT_TRUE(it == slice_range.end()) << "Expect empyt range";
}

///
/// slice specyfic test
///
TEST(coordinate_range, slice_range_input_validataion) {
    const Shape s{10, 10, 10};
    EXPECT_THROW(ov::coordinates::slice(s, {1}, {1}), std::domain_error);
    EXPECT_THROW(ov::coordinates::slice(s, s, {1}), std::domain_error);
    EXPECT_THROW(ov::coordinates::slice(s, {1}, s), std::domain_error);
    EXPECT_THROW(ov::coordinates::slice(s, s, s, {}), std::domain_error);
}

namespace {
Shape sliced_shape(const std::vector<size_t>& start_corner, const std::vector<size_t>& end_corner) {
    Shape s;
    std::transform(end_corner.begin(),
                   end_corner.end(),
                   start_corner.begin(),
                   std::back_inserter(s),
                   [](size_t e, size_t b) {
                       return e - b;
                   });

    return s;
}
Shape sliced_shape(const std::vector<size_t>& start_corner,
                   const std::vector<size_t>& end_corner,
                   const std::vector<size_t>& strides) {
    Shape s = sliced_shape(start_corner, end_corner);

    std::transform(s.begin(), s.end(), strides.begin(), s.begin(), [](size_t e, size_t s) {
        return (e + s - 1) / s;
    });

    return s;
}
}  // namespace

TEST(coordinate_range, slice_range_corner) {
    const Shape s{10, 10};
    const Coordinate source_start_corner{3, 3};
    const Coordinate source_end_corner{6, 6};
    const ExpectedOutput expected{{33, {3, 3}},
                                  {34, {3, 4}},
                                  {35, {3, 5}},
                                  {43, {4, 3}},
                                  {44, {4, 4}},
                                  {45, {4, 5}},
                                  {53, {5, 3}},
                                  {54, {5, 4}},
                                  {55, {5, 5}}};
    ASSERT_EQ(expected.size(), shape_size(sliced_shape(source_start_corner, source_end_corner)))
        << "check epxected data";

    auto expected_val = begin(expected);
    for (auto slice_range : ov::coordinates::slice(s, source_start_corner, source_end_corner)) {
        auto index = slice_range.begin_index;
        for (size_t i = 0; i < slice_range.element_number; index += slice_range.step, ++i) {
            EXPECT_EQ(index, expected_val->first);
            ++expected_val;
        }
    }
    EXPECT_TRUE(expected_val == end(expected))
        << "not all expected values return, (" << std::distance(expected_val, end(expected)) << " is missing)";
}

TEST(coordinate_range, slice_range_strides) {
    const Shape s{10, 10};
    const Coordinate source_start_corner{0, 0};
    const Coordinate source_end_corner{s};
    const Strides source_strides = Strides({2, 3});

    // clang-format off
    const ExpectedOutput expected{
        {0, {0, 0}},  {3, {0, 3}},  {6, {0, 6}},  {9, {0, 9}},
        {20, {2, 0}}, {23, {2, 3}}, {26, {2, 6}}, {29, {2, 9}},
        {40, {4, 0}}, {43, {4, 3}}, {46, {4, 6}}, {49, {4, 9}},
        {60, {6, 0}}, {63, {6, 3}}, {66, {6, 6}}, {69, {6, 9}},
        {80, {8, 0}}, {83, {8, 3}}, {86, {8, 6}}, {89, {8, 9}}};
    // clang-format on

    ASSERT_EQ(expected.size(), shape_size(sliced_shape(source_start_corner, source_end_corner, source_strides)))
        << "check epxected data";

    auto expected_val = begin(expected);
    for (auto slice_range : ov::coordinates::slice(s, source_start_corner, source_end_corner, source_strides)) {
        auto index = slice_range.begin_index;
        for (size_t i = 0; i < slice_range.element_number; index += slice_range.step, ++i) {
            EXPECT_EQ(index, expected_val->first);
            ++expected_val;
        }
    }
    EXPECT_TRUE(expected_val == end(expected))
        << "not all expected values return, (" << std::distance(expected_val, end(expected)) << " is missing)";
}

///
///
///    ReverseRange
///
///

TEST(coordinate_range, reverse_range_shape0d) {
    const Shape s;
    const AxisSet reverset_axis{};

    auto reverse_range = ov::coordinates::reverse(s, reverset_axis);
    auto it = reverse_range.begin();
    EXPECT_EQ(it, begin(reverse_range));
    auto v = *it;  // if it is not end it has to be dereferencable;
    (void)v;
    EXPECT_TRUE(++it == reverse_range.end());
}

TEST(coordinate_range, reverse_range_shape1d) {
    const Shape s{3};
    const AxisSet reverset_axis{};

    const ExpectedOutput expected{{0, {0}}, {1, {1}}, {2, {2}}};
    EXPECT_EQ(expected.size(), shape_size(s)) << "check epxected data";

    auto expected_val = begin(expected);
    for (auto reverse_range : ov::coordinates::reverse(s, reverset_axis)) {
        auto index = reverse_range.begin_index;
        ASSERT_EQ(reverse_range.direction, ov::coordinates::Direction::forward);
        for (size_t i = 0; i < reverse_range.element_number; index += reverse_range.step, ++i) {
            EXPECT_EQ(index, expected_val->first);
            ++expected_val;
        }
    }

    EXPECT_TRUE(expected_val == end(expected))
        << "not all expected values return, (" << std::distance(expected_val, end(expected)) << " is missing)";
}

TEST(coordinate_range, reverse_range_shape2d) {
    const Shape s{2, 3};
    const AxisSet reverset_axis{};

    // clang-format off
    const ExpectedOutput expected{
        {0, {0, 0}}, {1, {0, 1}}, {2, {0, 2}},
        {3, {1, 0}}, {4, {1, 1}}, {5, {1, 2}}};
    // clang-format on
    EXPECT_EQ(expected.size(), shape_size(s)) << "check epxected data";

    auto expected_val = begin(expected);
    for (auto reverse_range : ov::coordinates::reverse(s, reverset_axis)) {
        auto index = reverse_range.begin_index;
        ASSERT_EQ(reverse_range.direction, ov::coordinates::Direction::forward);
        for (size_t i = 0; i < reverse_range.element_number; index += reverse_range.step, ++i) {
            EXPECT_EQ(index, expected_val->first);
            ++expected_val;
        }
    }

    EXPECT_TRUE(expected_val == end(expected))
        << "not all expected values return, (" << std::distance(expected_val, end(expected)) << " is missing)";
}

TEST(coordinate_range, reverse_range_shape3d) {
    const Shape s{2, 3, 4};
    const AxisSet reverset_axis{};

    // clang-format off
    const ExpectedOutput expected{
         {0, {0, 0, 0}},  {1, {0, 0, 1}},  {2, {0, 0, 2}},  {3, {0, 0, 3}},
         {4, {0, 1, 0}},  {5, {0, 1, 1}},  {6, {0, 1, 2}},  {7, {0, 1, 3}},
         {8, {0, 2, 0}},  {9, {0, 2, 1}}, {10, {0, 2, 2}}, {11, {0, 2, 3}},
        {12, {1, 0, 0}}, {13, {1, 0, 1}}, {14, {1, 0, 2}}, {15, {1, 0, 3}},
        {16, {1, 1, 0}}, {17, {1, 1, 1}}, {18, {1, 1, 2}}, {19, {1, 1, 3}},
        {20, {1, 2, 0}}, {21, {1, 2, 1}}, {22, {1, 2, 2}}, {23, {1, 2, 3}}};
    // clang-format on
    EXPECT_EQ(expected.size(), shape_size(s)) << "check epxected data";

    auto expected_val = begin(expected);
    for (auto reverse_range : ov::coordinates::reverse(s, reverset_axis)) {
        auto index = reverse_range.begin_index;
        ASSERT_EQ(reverse_range.direction, ov::coordinates::Direction::forward);
        for (size_t i = 0; i < reverse_range.element_number; index += reverse_range.step, ++i) {
            EXPECT_EQ(index, expected_val->first);
            ++expected_val;
        }
    }

    EXPECT_TRUE(expected_val == end(expected))
        << "not all expected values return, (" << std::distance(expected_val, end(expected)) << " is missing)";
}

TEST(coordinate_range, reverse_range_zero_sized_axis) {
    const Shape s{2, 0, 4};

    auto reverse_range = ov::coordinates::reverse(s, {});
    auto it = reverse_range.begin();
    EXPECT_TRUE(it == reverse_range.end()) << "Expect empyt range";
}

///
/// reverse specyfic test
///
TEST(coordinate_range, reverse_range_input_validataion) {
    const Shape s{10, 10, 10};
    EXPECT_THROW(ov::coordinates::reverse(s, {10}), std::domain_error);
}

TEST(coordinate_range, reverse_range_2d) {
    const Shape s{3, 10};
    const AxisSet reverset_axis{1};

    // clang-format off
    const ExpectedOutput expected{
        {9, {0, 9}},   {8, {0, 8}},  {7, {0, 7}},  {6, {0, 6}},  {5, {0, 5}},  {4, {0, 4}},  {3, {0, 3}},  {2, {0, 2}},  {1, {0, 1}},  {0, {0, 0}},
        {19, {1, 9}}, {18, {1, 8}}, {17, {1, 7}}, {16, {1, 6}}, {15, {1, 5}}, {14, {1, 4}}, {13, {1, 3}}, {12, {1, 2}}, {11, {1, 1}}, {10, {1, 0}},
        {29, {2, 9}}, {28, {2, 8}}, {27, {2, 7}}, {26, {2, 6}}, {25, {2, 5}}, {24, {2, 4}}, {23, {2, 3}}, {22, {2, 2}}, {21, {2, 1}}, {20, {2, 0}}};
    // clang-format on
    auto expected_val = begin(expected);
    for (auto reverse_range : ov::coordinates::reverse(s, reverset_axis)) {
        auto index = reverse_range.begin_index;
        ASSERT_EQ(reverse_range.direction, ov::coordinates::Direction::reverse);
        for (size_t i = 0; i < reverse_range.element_number; index -= reverse_range.step, ++i) {
            EXPECT_EQ(index, expected_val->first);
            ++expected_val;
        }
    }

    EXPECT_TRUE(expected_val == end(expected))
        << "not all expected values return, (" << std::distance(expected_val, end(expected)) << " is missing)";
}

TEST(coordinate_range, reverse_1_range_3d) {
    const Shape s{3, 3, 3};
    const AxisSet reverset_axis{1};

    // clang-format off
    const ExpectedOutput expected{
         {6, {0, 2, 0}},  {7, {0, 2, 1}},  {8, {0, 2, 2}},
         {3, {0, 1, 0}},  {4, {0, 1, 1}},  {5, {0, 1, 2}},
         {0, {0, 0, 0}},  {1, {0, 0, 1}},  {2, {0, 0, 2}},

        {15, {1, 2, 0}}, {16, {1, 2, 1}}, {17, {1, 2, 2}},
        {12, {1, 1, 0}}, {13, {1, 1, 1}}, {14, {1, 1, 2}},
         {9, {1, 0, 0}}, {10, {1, 0, 1}}, {11, {1, 0, 2}},

        {24, {2, 2, 0}}, {25, {2, 2, 1}}, {26, {2, 2, 2}},
        {21, {2, 1, 0}}, {22, {2, 1, 1}}, {23, {2, 1, 2}},
        {18, {2, 0, 0}}, {19, {2, 0, 1}}, {20, {2, 0, 2}}};
    // clang-format on

    auto expected_val = begin(expected);
    for (auto reverse_range : ov::coordinates::reverse(s, reverset_axis)) {
        auto index = reverse_range.begin_index;
        ASSERT_EQ(reverse_range.direction, ov::coordinates::Direction::forward);
        for (size_t i = 0; i < reverse_range.element_number; index += reverse_range.step, ++i) {
            EXPECT_EQ(index, expected_val->first);
            ++expected_val;
        }
    }

    EXPECT_TRUE(expected_val == end(expected))
        << "not all expected values return, (" << std::distance(expected_val, end(expected)) << " is missing)";
}

TEST(coordinate_range, reverse_2_range_3d) {
    const Shape s{3, 3, 3};
    const AxisSet reverset_axis{1, 2};

    // clang-format off
    const ExpectedOutput expected{
         {8, {0, 2, 2}},  {7, {0, 2, 1}},  {6, {0, 2, 0}},
         {5, {0, 1, 2}},  {4, {0, 1, 1}},  {3, {0, 1, 0}},
         {2, {0, 0, 2}},  {1, {0, 0, 1}},  {0, {0, 0, 0}},

        {17, {1, 2, 2}}, {16, {1, 2, 1}}, {15, {1, 2, 0}},
        {14, {1, 1, 2}}, {13, {1, 1, 1}}, {12, {1, 1, 0}},
        {11, {1, 0, 2}}, {10, {1, 0, 1}},  {9, {1, 0, 0}},

        {26, {2, 2, 2}}, {25, {2, 2, 1}}, {24, {2, 2, 0}},
        {23, {2, 1, 2}}, {22, {2, 1, 1}}, {21, {2, 1, 0}},
        {20, {2, 0, 2}}, {19, {2, 0, 1}}, {18, {2, 0, 0}}};
    // clang-format on

    auto expected_val = begin(expected);
    for (auto reverse_range : ov::coordinates::reverse(s, reverset_axis)) {
        auto index = reverse_range.begin_index;
        ASSERT_EQ(reverse_range.direction, ov::coordinates::Direction::reverse);
        for (size_t i = 0; i < reverse_range.element_number; index -= reverse_range.step, ++i) {
            EXPECT_EQ(index, expected_val->first);
            ++expected_val;
        }
    }

    EXPECT_TRUE(expected_val == end(expected))
        << "not all expected values return, (" << std::distance(expected_val, end(expected)) << " is missing)";
}
