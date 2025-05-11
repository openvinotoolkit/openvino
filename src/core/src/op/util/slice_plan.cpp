// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/slice_plan.hpp"

#include <algorithm>

#include "openvino/core/except.hpp"

namespace ov {
namespace op {
namespace util {

SlicePlan make_slice_plan(const Shape& input_shape,
                          const std::vector<int64_t>& begins,
                          const std::vector<int64_t>& ends,
                          const std::vector<int64_t>& strides,
                          const AxisSet& lower_bounds_mask,
                          const AxisSet& upper_bounds_mask,
                          const AxisSet& new_axis_mask,
                          const AxisSet& shrink_axis_mask,
                          const AxisSet& ellipsis_mask) {
    OPENVINO_ASSERT(begins.size() == ends.size());
    OPENVINO_ASSERT(ends.size() == strides.size());
    size_t num_slice_indices = begins.size();

    size_t num_real_axes = 0;
    size_t num_shrink_axes = 0;
    size_t num_new_axes = 0;
    bool ellipsis_found = false;

    // Make a pass over the original slices to make sure there is at most one
    // ellipsis, and to count up the number of shrink axes, the number of
    // "newaxis"es, and the number of "real" axes (axes that are not newaxis
    // and are not the ellipsis).
    for (size_t i = 0; i < num_slice_indices; i++) {
        if (ellipsis_mask.count(i)) {
            OPENVINO_ASSERT(!ellipsis_found);
            ellipsis_found = true;
        } else if (new_axis_mask.count(i)) {
            num_new_axes++;
        } else {
            if (shrink_axis_mask.count(i)) {
                num_shrink_axes++;
            }
            num_real_axes++;
        }
    }

    OPENVINO_ASSERT(num_real_axes <= input_shape.size(),
                    "num_real_axes=",
                    num_real_axes,
                    ", input_shape=",
                    input_shape);

    // Figure out how many axes need to be inserted when the ellipsis (which
    // may be an implicit ellipsis at the end) is expanded.
    size_t ellipsis_size = input_shape.size() - num_real_axes;

    // Initialize our slice plan.
    SlicePlan p;
    p.begins = std::vector<int64_t>(num_real_axes + ellipsis_size);
    p.ends = std::vector<int64_t>(num_real_axes + ellipsis_size);
    p.strides = std::vector<int64_t>(num_real_axes + ellipsis_size);
    p.reshape_in_shape = Shape(num_real_axes + ellipsis_size);
    p.reshape_out_shape = Shape(num_new_axes + num_real_axes + ellipsis_size - num_shrink_axes);
    p.reverse_axes = AxisSet{};

    // Begin a maddeningly delicate loop to desugar the original slice.
    //
    // * i_in is iterating over the axes of the input shape, which are also the axes of
    //     p.reshape_in_shape.
    // * i_out is iterating over the axes of p.reshape_out_shape
    size_t i_in = 0;
    size_t i_out = 0;

    // If no actual ellipsis exists, there is an "implicit" one at the end,
    // which we will handle after the loop. So the logic is wrapped up here,
    // allowing it to be used both during and after the loop.
    auto expand_ellipsis = [&]() {
        for (size_t i = 0; i < ellipsis_size; i++) {
            p.begins[i_in] = 0;
            p.ends[i_in] = int64_t(input_shape[i_in]);
            p.strides[i_in] = 1;
            p.reshape_in_shape[i_in] = input_shape[i_in];
            p.reshape_out_shape[i_out] = input_shape[i_in];

            i_in++;
            i_out++;
        }
    };

    for (size_t i = 0; i < num_slice_indices; i++) {
        // If this is a "newaxis", then reshape_out_shape will have a 1 here,
        // but reshape_in_shape will not.
        if (new_axis_mask.count(i)) {
            p.reshape_out_shape[i_out] = 1;
            i_out++;
        }
        // If this is a "shrunken" axis, then reshape_in_shape will have a 1
        // here, but reshape_out_shape will not.
        else if (shrink_axis_mask.count(i)) {
            int64_t begin = begins[i];

            // Note that clipping is not used for "shrunken" axes: an
            // out-of-bounds index is an error.
            OPENVINO_ASSERT(begin >= -(int64_t(input_shape[i_in])) && begin < int64_t(input_shape[i_in]));

            if (begin < 0) {
                begin += int64_t(input_shape[i_in]);
            }
            p.begins[i_in] = begin;
            p.ends[i_in] = begin + 1;
            p.strides[i_in] = 1;
            p.reshape_in_shape[i_in] = 1;
            i_in++;
        }
        // If this is the ellipsis, expand it.
        else if (ellipsis_mask.count(i)) {
            expand_ellipsis();
        }
        // In other cases, we have a nice, ordinary (begin:end:stride) slice.
        // We need to adjust for begin/end being masked, and begin/end/stride
        // being negative or out of bounds.
        else {
            bool is_reverse = strides[i] < 0;

            // Adjust the beginning for from-the-right indexing, and clip.
            int64_t real_begin = begins[i];
            if (lower_bounds_mask.count(i)) {
                real_begin = (is_reverse ? int64_t(input_shape[i_in] - 1) : 0);
            } else if (real_begin < 0) {
                real_begin += int64_t(input_shape[i_in]);
            }
            int64_t max_real_begin = int64_t(input_shape[i_in]) - (is_reverse ? 1 : 0);
            real_begin = std::max(int64_t(0), std::min(max_real_begin, real_begin));

            // Adjust the ending for from-the-right indexing, and clip.
            int64_t real_end = ends[i];
            if (upper_bounds_mask.count(i)) {
                real_end = (is_reverse ? -1 : int64_t(input_shape[i_in]));
            } else if (real_end < 0) {
                real_end += int64_t(input_shape[i_in]);
            }
            int64_t min_real_end = (is_reverse ? -1 : 0);
            real_end = std::max(min_real_end, std::min(int64_t(input_shape[i_in]), real_end));

            // Ensure stride is not zero, and adjust it for backwards slicing.
            OPENVINO_ASSERT(strides[i] != 0);
            int64_t real_stride = std::abs(strides[i]);

            // Adjust for reversal if needed. This isn't quite as simple as swapping begin and
            // end, due to striding; we have to adjust the end point to be the _actual_ leftmost
            // element, in cases where the stride does not evenly divide the span between begin
            // and end.
            if (is_reverse) {
                real_end += std::max(int64_t(0), real_begin - real_end - 1) % real_stride;
                std::swap(real_begin, real_end);
                real_begin++;
                real_end++;
                p.reverse_axes.insert(i_out);
            }

            // ov slice op does not like it when end < begin, so we truncate for that case here.
            if (real_end < real_begin) {
                real_end = real_begin;
            }

            // Compute output dimension.
            size_t dim = (real_end <= real_begin ? 0 : size_t(real_end - real_begin - 1) / size_t(real_stride) + 1);
            p.reshape_in_shape[i_in] = dim;
            p.reshape_out_shape[i_out] = dim;

            auto slice_size = real_end - real_begin;
            if (slice_size > 0 && real_stride > slice_size)
                real_stride = slice_size;
            if (real_stride == slice_size) {
                real_end = real_begin + 1;
                real_stride = 1;
            }

            // Set up the begin/end/stride.
            p.begins[i_in] = real_begin;
            p.ends[i_in] = real_end;
            p.strides[i_in] = real_stride;

            i_in++;
            i_out++;
        }
    }

    // If there was no ellipsis explicitly given, there is an implicit one at
    // the end (it might encompass zero axes, but that's fine).
    if (!ellipsis_found) {
        expand_ellipsis();
    }
    return p;
}

bool SlicePlan::operator==(const SlicePlan& other) const {
    bool equal = true;
    equal &= begins == other.begins;
    equal &= ends == other.ends;
    equal &= strides == other.strides;
    equal &= reshape_in_shape == other.reshape_in_shape;
    equal &= reshape_out_shape == other.reshape_out_shape;
    equal &= reverse_axes == other.reverse_axes;

    return equal;
}

bool SlicePlan::operator!=(const SlicePlan& other) const {
    return !(*this == other);
}
}  // namespace util
}  // namespace op
}  // namespace ov
