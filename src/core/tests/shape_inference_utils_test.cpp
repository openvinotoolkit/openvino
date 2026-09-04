// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <limits>

#include "openvino/op/reverse.hpp"
#include "slice_shape_inference_utils.hpp"
#include "utils.hpp"

TEST(shape_inference_utils_test, get_input_bounds_not_valid_port) {
    ov::op::v1::Reverse dummy_op;
    const size_t not_valid_port = 100;
    const ov::ITensorAccessor& ta = ov::make_tensor_accessor();

    const auto ret = ov::op::get_input_bounds<ov::PartialShape, int64_t>(&dummy_op, not_valid_port, ta);
    ASSERT_FALSE(ret);
}

TEST(shape_inference_utils_test, is_size_preserving_slice) {
    // True only if the slice yields exactly L elements for every length L within the dimension interval and every
    // start/stop value within the bounds.
    using ov::op::slice::Bounds;
    constexpr auto i64_max = std::numeric_limits<int64_t>::max();
    constexpr auto i64_min = std::numeric_limits<int64_t>::min();

    struct Row {
        ov::Dimension dim;
        Bounds start;
        Bounds stop;
        int64_t step;
        bool expected;
    };
    const auto inf = ov::Dimension(1, -1);
    const std::vector<Row> rows{
        {inf, {0, 0}, {i64_max, i64_max}, 1, true},
        {inf, {0, 0}, {i64_max, i64_max}, 2, false},
        {inf, {0, 0}, {i64_max, i64_max}, 3, false},
        {inf, {0, 0}, {i64_max, i64_max}, -1, false},
        {inf, {i64_min, i64_min}, {i64_max, i64_max}, 1, true},
        {inf, {i64_max, i64_max}, {i64_min, i64_min}, -1, true},
        {inf, {-1, -1}, {i64_min, i64_min}, -1, true},
        {inf, {1, 1}, {i64_max, i64_max}, 1, false},
        {inf, {0, 0}, {2147483647, 2147483647}, 1, false},
        {inf, {i64_min, 0}, {i64_max, i64_max}, 1, false},
        {inf, {0, 0}, {1, i64_max}, 1, false},
        {ov::Dimension(10), {0, 0}, {10, 10}, 1, true},
        {ov::Dimension(10), {0, 0}, {9, 9}, 1, false},
        {ov::Dimension(10), {-20, -20}, {20, 20}, 1, true},
        {ov::Dimension(10), {9, 9}, {-11, -11}, -1, true},
        {ov::Dimension(4, 8), {0, 0}, {8, 8}, 1, true},
        {ov::Dimension(4, 8), {0, 0}, {6, 6}, 1, false},
        {ov::Dimension(4, 8), {0, 0}, {4, 8}, 1, false},
        {ov::Dimension(2, 5), {-5, -5}, {5, 5}, 1, true},
        {ov::Dimension(2, 5), {-5, -3}, {5, 5}, 1, false},
        {ov::Dimension(2, 5), {0, 0}, {5, 5}, -1, false},
        {ov::Dimension(2, 5), {4, 4}, {-6, -6}, -1, true},
        {ov::Dimension(2, 5), {3, 3}, {-6, -6}, -1, false},
        // lengths 0 and 1 are trivially preserved by any step: the predicate is conservative here
        {ov::Dimension(0, 1), {0, 0}, {i64_max, i64_max}, 2, false},
    };

    for (size_t i = 0; i < rows.size(); ++i) {
        const auto& r = rows[i];
        EXPECT_EQ(ov::op::slice::is_size_preserving_slice(r.dim, r.start, r.stop, r.step), r.expected) << "row " << i;
    }
}
