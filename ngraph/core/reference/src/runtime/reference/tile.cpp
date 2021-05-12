// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
        pitch.resize(dims.size() - 1);
        std::partial_sum(
            dims.rbegin(), dims.rend() - 1, pitch.rbegin(), std::multiplies<int64_t>());
        pitch.push_back(1);
        return pitch;
    }
} // namespace

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

    std::vector<size_t> indices(in_shape_expanded.size() - 1, 0);
    size_t axis = indices.size();

    // Copy and repeat data for innermost axis as many times as described in the repeats parameter
    while (axis <= indices.size())
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

        // Copy and repeat data for other axes as many times as described in the repeats parameter
        while (axis-- != 0)
        {
            if (++indices[axis] != in_shape_expanded[axis])
            {
                axis = indices.size();
                break;
            }
            indices[axis] = 0;

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
