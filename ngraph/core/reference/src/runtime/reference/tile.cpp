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

#include <cmath>
#include <stdio.h>

#include "ngraph/check.hpp"
#include "ngraph/runtime/reference/tile.hpp"

using namespace ngraph;

bool Increment(std::vector<int64_t>& indices, size_t& axis, const Shape& input_shape, bool& run)
{
    if (axis-- == 0)
    {
        run = false;
        return false;
    }
    
    if (++indices[axis] != input_shape[axis])
    {
        axis = indices.size();
        return false;
    }

    indices[axis] = 0;
    return true;
}

std::vector<int64_t> createPitch(const Shape& dims)
{
    std::vector<int64_t> pitch;
    auto tensor_rank = dims.size();
    /*
    if (tensor_rank == 1)
    {
        return pitch;
    }
    */
    for (int i = 1; i < tensor_rank - 1; i++)
    {
        int64_t val = std::accumulate(dims.begin() + 1, dims.end(), 0);
        pitch.push_back(val);
    }

    pitch.push_back(1);    

    return pitch;
}

void runtime::reference::tile(
    const char* arg, char* out, const Shape& in_shape, const Shape& out_shape, size_t elem_size, std::vector<int64_t> repeats)
{
    
    Shape in_shape_expanded(in_shape);
    in_shape_expanded.insert(in_shape_expanded.begin(), out_shape.size() - in_shape.size(), 1);
    int input_rank = in_shape_expanded.size();
    std::vector<int64_t> repeats;
    size_t block_size = 0;
    int64_t num_repeats = 0;
    const char* copy = nullptr;
    const int64_t innermost_dim = in_shape_expanded[input_rank - 1];
    std::vector<int64_t> indices{input_rank - 1, 0};
    std::vector<int64_t> output_pitches;
    size_t axis = indices.size();
    bool run = true;
    
    output_pitches = createPitch(out_shape);

    while(run)
    {
        block_size = innermost_dim * elem_size;
        memcpy(out, arg, block_size);
        out += block_size;
        arg += block_size;

        copy = out - block_size;
        num_repeats = repeats[input_rank - 1] - 1;
        for (int64_t repeat = 0; repeat < num_repeats; ++repeat)
        {
            memcpy(out, copy, block_size);
        } 

        while(Increment(indices, axis, in_shape, run))
        {
            ptrdiff_t pitch = output_pitches[axis] * in_shape_expanded[axis];
            block_size = pitch * elem_size;
            copy = out - block_size;
            num_repeats = repeats[axis] - 1;
            for (int64_t repeat = 0; repeat < num_repeats; repeat++)
            {
                memcpy(out, copy, block_size);
                out += block_size;
            }
        }
    }
}

