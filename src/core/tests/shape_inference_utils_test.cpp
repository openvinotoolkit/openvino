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

using ov::op::slice::Bounds;

// dim, start, stop, step, expected
using IsSizePreservingSliceParams = std::tuple<ov::Dimension, Bounds, Bounds, int64_t, bool>;

class IsSizePreservingSliceTest : public testing::TestWithParam<IsSizePreservingSliceParams> {};

// True only if the slice yields exactly L elements for every length L within the dimension interval and every
// start/stop value within the bounds.
TEST_P(IsSizePreservingSliceTest, is_size_preserving_slice) {
    const auto& [dim, start, stop, step, expected] = GetParam();

    EXPECT_EQ(ov::op::slice::is_size_preserving_slice(dim, start, stop, step), expected);
}

namespace {
constexpr auto i64_max = std::numeric_limits<int64_t>::max();
constexpr auto i64_min = std::numeric_limits<int64_t>::min();
}  // namespace

INSTANTIATE_TEST_SUITE_P(
    shape_inference_utils_test,
    IsSizePreservingSliceTest,
    testing::Values(
        IsSizePreservingSliceParams{ov::Dimension::dynamic(), {0, 0}, {i64_max, i64_max}, 1, true},
        IsSizePreservingSliceParams{ov::Dimension::dynamic(), {0, 0}, {i64_max, i64_max}, 2, false},
        IsSizePreservingSliceParams{ov::Dimension::dynamic(), {0, 0}, {i64_max, i64_max}, 3, false},
        IsSizePreservingSliceParams{ov::Dimension::dynamic(), {0, 0}, {i64_max, i64_max}, -1, false},
        IsSizePreservingSliceParams{ov::Dimension::dynamic(), {i64_min, i64_min}, {i64_max, i64_max}, 1, true},
        IsSizePreservingSliceParams{ov::Dimension::dynamic(), {i64_max, i64_max}, {i64_min, i64_min}, -1, true},
        IsSizePreservingSliceParams{ov::Dimension::dynamic(), {-1, -1}, {i64_min, i64_min}, -1, true},
        IsSizePreservingSliceParams{ov::Dimension::dynamic(), {1, 1}, {i64_max, i64_max}, 1, false},
        IsSizePreservingSliceParams{ov::Dimension::dynamic(), {0, 0}, {2147483647, 2147483647}, 1, false},
        IsSizePreservingSliceParams{ov::Dimension::dynamic(), {i64_min, 0}, {i64_max, i64_max}, 1, false},
        IsSizePreservingSliceParams{ov::Dimension::dynamic(), {0, 0}, {1, i64_max}, 1, false},
        IsSizePreservingSliceParams{ov::Dimension(10), {0, 0}, {10, 10}, 1, true},
        IsSizePreservingSliceParams{ov::Dimension(10), {0, 0}, {9, 9}, 1, false},
        IsSizePreservingSliceParams{ov::Dimension(10), {-20, -20}, {20, 20}, 1, true},
        IsSizePreservingSliceParams{ov::Dimension(10), {9, 9}, {-11, -11}, -1, true},
        IsSizePreservingSliceParams{ov::Dimension(4, 8), {0, 0}, {8, 8}, 1, true},
        IsSizePreservingSliceParams{ov::Dimension(4, 8), {0, 0}, {6, 6}, 1, false},
        IsSizePreservingSliceParams{ov::Dimension(4, 8), {0, 0}, {4, 8}, 1, false},
        IsSizePreservingSliceParams{ov::Dimension(2, 5), {-5, -5}, {5, 5}, 1, true},
        IsSizePreservingSliceParams{ov::Dimension(2, 5), {-5, -3}, {5, 5}, 1, false},
        IsSizePreservingSliceParams{ov::Dimension(2, 5), {0, 0}, {5, 5}, -1, false},
        IsSizePreservingSliceParams{ov::Dimension(2, 5), {4, 4}, {-6, -6}, -1, true},
        IsSizePreservingSliceParams{ov::Dimension(2, 5), {3, 3}, {-6, -6}, -1, false},
        // lengths 0 and 1 are trivially preserved by any step: the predicate is conservative here
        IsSizePreservingSliceParams{ov::Dimension(0, 1), {0, 0}, {i64_max, i64_max}, 2, false},
        // a start/stop bound one away from the INT64_MIN/INT64_MAX sentinel still clips for every length up
        // to max_length (max_length - 1 away from the sentinel in the opposite direction of the check)
        IsSizePreservingSliceParams{ov::Dimension::dynamic(), {i64_min + 1, i64_min + 1}, {i64_max, i64_max}, 1, true},
        IsSizePreservingSliceParams{ov::Dimension::dynamic(),
                                    {i64_max - 1, i64_max - 1},
                                    {i64_min, i64_min},
                                    -1,
                                    true}));
