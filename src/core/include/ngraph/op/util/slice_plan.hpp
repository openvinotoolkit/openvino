// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if !defined(IN_OV_COMPONENT) && !defined(NGRAPH_LEGACY_HEADER_INCLUDED)
#    define NGRAPH_LEGACY_HEADER_INCLUDED
#    ifdef _MSC_VER
#        pragma message( \
            "The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    else
#        warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    endif
#endif

#include <set>

#include "ngraph/axis_set.hpp"
#include "ngraph/shape.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START
namespace ngraph {
//
// In various places, like ConstantFolding, it is
// useful to transform DynSlice by converting it to a sequence of ops:
//
//      Slice    (to do the basic slicing)
//        |
//        v
//     Reshape   (non-transposing, to handle shrinks)
//        |
//        v
//     Reverse   (to emulate backwards stride)
//
// (The Reshape, Reverse, or both may be omitted if they would just be
// identities.)
//
// A SlicePlan is used to collect parameters for these ops.
//
// This class is moved to dev API
struct NGRAPH_API_DEPRECATED NGRAPH_API SlicePlan {
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

NGRAPH_API_DEPRECATED SlicePlan NGRAPH_API make_slice_plan(const Shape& input_shape,
                                                           const std::vector<int64_t>& begins,
                                                           const std::vector<int64_t>& ends,
                                                           const std::vector<int64_t>& strides,
                                                           const AxisSet& lower_bounds_mask,
                                                           const AxisSet& upper_bounds_mask,
                                                           const AxisSet& new_axis_mask,
                                                           const AxisSet& shrink_axis_mask,
                                                           const AxisSet& ellipsis_mask);
}  // namespace ngraph

using ngraph::make_slice_plan;
NGRAPH_SUPPRESS_DEPRECATED_END
