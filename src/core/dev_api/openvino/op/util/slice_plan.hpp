// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/axis_set.hpp"
#include "openvino/core/shape.hpp"

namespace ov {
namespace op {
namespace util {

/**
 * @brief A collection of parameters for advanced slicing
 * @details In various places, like ConstantFolding, it is useful to transform DynSlice by converting it to a sequence
 * of ops:
 *
 *      Slice    (to do the basic slicing)
 *        |
 *        v
 *     Reshape   (non-transposing, to handle shrinks)
 *        |
 *        v
 *     Reverse   (to emulate backwards stride)
 *
 * (The Reshape, Reverse, or both may be omitted if they would just be identities.)
 *
 * A SlicePlan is used to collect parameters for these ops.
 **/
struct OPENVINO_API SlicePlan {
    // Parameters for the Slice
    std::vector<int64_t> begins;
    std::vector<int64_t> ends;
    std::vector<int64_t> strides;

    // Shapes coming into, and going out of, the Reshape.
    Shape reshape_in_shape;
    Shape reshape_out_shape;

    // Parameters for the Reverse
    AxisSet reverse_axes;

    bool operator==(const SlicePlan& other) const;
    bool operator!=(const SlicePlan& other) const;
};

/**
 * @brief Prepares slice plan for strided slicing
 **/
SlicePlan OPENVINO_API make_slice_plan(const Shape& input_shape,
                                       const std::vector<int64_t>& begins,
                                       const std::vector<int64_t>& ends,
                                       const std::vector<int64_t>& strides,
                                       const AxisSet& lower_bounds_mask,
                                       const AxisSet& upper_bounds_mask,
                                       const AxisSet& new_axis_mask,
                                       const AxisSet& shrink_axis_mask,
                                       const AxisSet& ellipsis_mask);
}  // namespace util
}  // namespace op
}  // namespace ov
