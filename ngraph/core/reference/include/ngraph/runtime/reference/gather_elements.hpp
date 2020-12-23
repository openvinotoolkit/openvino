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

#include <algorithm>
#include <cassert>
#include <numeric>

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T, typename U>
            void gather_elements(const T* data,
                                 const U* indices,
                                 T* out,
                                 const Shape& data_shape,
                                 const Shape& indices_shape,
                                 const Shape& out_shape,
                                 int64_t axis)
            {
                /*
                 K, N, M - let it be depth, row and column sizes of a 3D tensor
                 k, n, m - corresponding indices
                 M*(N*k + n) + m
                 M*N*k + M*n + m   <-- index after flattening of a 3D array

                 P, K, N, M - p, k, n, m
                 M*(N*(K*p + k) + n) + m
                 M*N*K*p + M*N*k + M*n + m   <-- index after flattening of a 4D array
                */

                // in 1D case results can be achieved without additional calculations
                if (axis < 0)
                {
                    axis += data_shape.size();
                }

                if (data_shape.size() == 1)
                {
                    for (int64_t i = 0; i < indices_shape[0]; i++)
                    {
                        if (indices[i] > data_shape[0])
                        {
                            throw std::domain_error{
                                "indices values of GatherElement exceed data size"};
                        }
                        out[i] = data[indices[i]];
                    }
                    return;
                }

                int64_t axis_mul = 1; // axis_mul = M*N*K in 3D case if axis = 0
                for (int64_t i = axis + 1; i < data_shape.size(); i++)
                {
                    axis_mul *= data_shape[i];
                }

                int64_t data_idx;
                for (int64_t i = 0; i < ngraph::shape_size(indices_shape); i++)
                {
                    data_idx = i - axis_mul * (((i / axis_mul) % data_shape[axis]) - indices[i]);
                    if (data_idx > ngraph::shape_size(data_shape))
                    {
                        throw std::domain_error{"indices values of GatherElement exceed data size"};
                    }
                    out[i] = data[data_idx];
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
