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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>

#include "ngraph/check.hpp"
#include "ngraph/runtime/reference/tile.hpp"

using namespace ngraph;

namespace
{
    /// \brief For each axis calculates the product of inner axes
    /// If dims has shape (2, 3, 4) then for 2 (first axis) the inner axes would be (3, 4)
    /// and for 3 (second axis) it would be (4)
    /// If dims has shape(2, 3, 4) then the output vector would be (3 * 4, 4, 1)
    /// The outermost axis is not used. For innermost axis it is always 1.
    /// \param[in] dims Shape of the output
    ///
    /// \return Vector containing calculated values for each axis.
    std::vector<int64_t> create_pitches(const Shape& dims)
    {
        std::vector<int64_t> pitch;
        pitch.resize(dims.size());
        std::partial_sum(dims.rbegin(), dims.rend() - 1, pitch.begin(), std::multiplies<int64_t>());
        std::reverse(pitch.begin(), pitch.end() - 1);

        pitch.back() = 1;
        return pitch;
    }

    bool calculate_next_axis(std::vector<int64_t>& indices, int64_t& axis, const Shape& shape)
    {
        if (axis-- == 0)
        {
            return false;
        }

        if (++indices[axis] != shape[axis])
        {
            axis = indices.size();
            return false;
        }

        indices[axis] = 0;
        return true;
    }
}

void runtime::reference::tile(const char* arg,
                              char* out,
                              const Shape& in_shape,
                              const Shape& out_shape,
                              const size_t elem_size,
                              const std::vector<int64_t>& repeats)
{
    Shape in_shape_expanded(in_shape);
    in_shape_expanded.insert(in_shape_expanded.begin(), out_shape.size() - in_shape.size(), 1);
    size_t block_size = 0;
    int64_t num_repeats = 0;
    const int input_rank = in_shape_expanded.size();
    const int64_t last_dim = in_shape_expanded[input_rank - 1];
    const std::vector<int64_t> pitches = create_pitches(out_shape);
    const char* copy = nullptr;

    std::vector<int64_t> indices(in_shape_expanded.size() - 1, 0);
    int64_t axis = indices.size();

    while (axis >= 0)
    {
        block_size = last_dim * elem_size;
        memcpy(out, arg, block_size);
        out += block_size;
        arg += block_size;

        copy = out - block_size;
        num_repeats = repeats[input_rank - 1] - 1;
        for (int64_t i = 0; i < num_repeats; ++i)
        {
            memcpy(out, copy, block_size);
            out += block_size;
        }

        while (calculate_next_axis(indices, axis, in_shape_expanded))
        {
            ptrdiff_t pitch = pitches[axis] * in_shape_expanded[axis];
            block_size = pitch * elem_size;
            copy = out - block_size;
            num_repeats = repeats[axis] - 1;
            for (int64_t i = 0; i < num_repeats; i++)
            {
                memcpy(out, copy, block_size);
                out += block_size;
            }
        }
    }
}
