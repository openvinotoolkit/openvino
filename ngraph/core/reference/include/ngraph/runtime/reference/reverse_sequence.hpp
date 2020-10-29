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

#include <cmath>
#include <numeric>

#include "ngraph/coordinate_transform.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T, typename U>
            void reverse_sequence(const T* arg,
                                  T* out,
                                  const Shape& arg_shape,
                                  size_t batch_axis,
                                  size_t sequence_axis,
                                  const U* sequence_lengths)
            {
                CoordinateTransform input_transform(arg_shape);
                for (const Coordinate& in_coord : input_transform)
                {
                    size_t batch_index = in_coord[batch_axis];
                    auto orig_seq_index = static_cast<size_t>(sequence_lengths[batch_index]);

                    if (orig_seq_index > arg_shape.at(sequence_axis))
                    {
                        throw ngraph_error(
                            "One of the elements of sequence lengths is greater than sequence axis "
                            "dimension");
                    }

                    if (orig_seq_index == 0)
                    {
                        orig_seq_index = 1;
                    }

                    size_t sequence_index = in_coord[sequence_axis] < orig_seq_index
                                                ? orig_seq_index - in_coord[sequence_axis] - 1
                                                : in_coord[sequence_axis];

                    // make a copy of in_coord and update sequence_index
                    Coordinate out_coord = in_coord;
                    out_coord[sequence_axis] = sequence_index;
                    out[input_transform.index(out_coord)] = arg[input_transform.index(in_coord)];
                }
            }
        }
    }
}
