//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <set>

#include "ngraph/axis_set.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    //
    // In various places, like ConstantFolding and DynElimination, it is
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
    struct NGRAPH_API SlicePlan
    {
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

    SlicePlan NGRAPH_API make_slice_plan(const Shape& input_shape,
                                         const std::vector<int64_t>& begins,
                                         const std::vector<int64_t>& ends,
                                         const std::vector<int64_t>& strides,
                                         const AxisSet& lower_bounds_mask,
                                         const AxisSet& upper_bounds_mask,
                                         const AxisSet& new_axis_mask,
                                         const AxisSet& shrink_axis_mask,
                                         const AxisSet& ellipsis_mask);
}
