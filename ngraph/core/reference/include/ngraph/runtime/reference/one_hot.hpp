//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

#include "ngraph/coordinate_transform.hpp"
#include "ngraph/shape_util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            // Below is an old version to delete
            template <typename INDICES_TYPE, typename OUTPUT_TYPE>
            void one_hot(const INDICES_TYPE* arg,
                         OUTPUT_TYPE* out,
                         const Shape& in_shape,
                         const Shape& out_shape,
                         size_t one_hot_axis,
                         const OUTPUT_TYPE on_value,
                         const OUTPUT_TYPE off_value)
            {
                // Step 1: Set off_value to the output.
                CoordinateTransform output_transform(out_shape);

                for (const Coordinate& output_coord : output_transform)
                {
                    out[output_transform.index(output_coord)] = off_value;
                }

                // Step 2: Write off_value at needed positions, throwing exceptions when invalid
                // conditions are encountered.
                CoordinateTransform input_transform(in_shape);

                for (const Coordinate& input_coord : input_transform)
                {
                    INDICES_TYPE val = arg[input_transform.index(input_coord)];

                    if (std::floor(val) < val || std::floor(val) > val)
                    {
                        continue;
                    }

                    size_t one_hot_pos = static_cast<size_t>(val);

                    if (one_hot_pos >= out_shape[one_hot_axis])
                    {
                        continue;
                    }

                    Coordinate one_hot_coord = inject(input_coord, one_hot_axis, one_hot_pos);

                    out[output_transform.index(one_hot_coord)] = on_value;
                }
            }
            /*

            template <typename T>
            void one_hot(const T* arg,
                         T* out,
                         const Shape& in_shape,
                         const Shape& out_shape,
                         size_t one_hot_axis)
            {
                const T on_value = 1;
                const T off_value = 0;
                one_hot<T, T>(arg, out, in_shape, out_shape, one_hot_axis, on_value, off_value);
            }
             */
            // Below is a new version to keep

            template <typename INPUT_TYPE>
            void one_hot(const INPUT_TYPE* indices,
                         char* out,
                         const Shape& indices_shape,
                         const Shape& out_shape,
                         size_t out_elem_size,
                         size_t one_hot_axis,
                         const char* on_value,
                         const char* off_value) {

                const size_t num_ind = shape_size(indices_shape);
                const size_t depth = out_shape[one_hot_axis];

                // Step 1: Set off_value to the output.
                for(auto p=out; p < out + num_ind * depth * out_elem_size; p += out_elem_size)
                    std::copy(off_value, off_value + out_elem_size, p);
                // Number of elements between one-hot values in the output memory layout
                size_t inner_block = 1;
                for(auto i=one_hot_axis; i < indices_shape.size(); ++i)
                    inner_block *= indices_shape[i];
                // Step 2: Write on_value at needed positions, throwing exceptions when invalid
                // conditions are encountered.
                for(auto outer_i=0; outer_i < num_ind; outer_i+=inner_block){
                    for(auto inner_i=0; inner_i < inner_block; inner_i++){
                        auto input_val = indices[outer_i + inner_i];
                        // Enable the check, if negative indices are not allowed
                        //NGRAPH_CHECK(input_val >= 0, "Only non-negative input indices are allowed, got ", input_val);
                        if ( (input_val >= 0) && (input_val < depth) ) {
                            auto output_offset = (outer_i * depth + inner_i + input_val * inner_block) * out_elem_size;
                            std::copy(on_value, on_value + out_elem_size, out + output_offset);
                        }
                    }
                }
            }
        }
    }
}
