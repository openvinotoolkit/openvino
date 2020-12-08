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

//#include "ngraph/coordinate_transform.hpp"

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
                        int64_t axis){
               /*
                K, N, M - depth row column
                M*n + m
                M*(N*k + n) + m
                P K N M - p k n m
                M*(N*(K*p + k) + n) + m
                M*N*K*p + M*N*k + M*n + m
               */

                size_t count = 1;
                for (size_t i = 0; i < indices_shape.size(); i++){
                    count *= indices_shape[i];
                }

                size_t axis_size = 1;  // axis_size = M*N*K if axis = 0
                for (size_t i = data_shape.size() - 1; i > axis; i--){
                    axis_size *= data_shape[i];
                }

                size_t data_idx;
                for (size_t i = 0; i < count; i++){
                    data_idx = i - axis_size * ((i / data_shape[axis]) - indices[i]);
                    out[i] = data[data_idx];
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
