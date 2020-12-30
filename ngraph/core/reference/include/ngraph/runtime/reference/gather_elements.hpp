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

#include "ngraph/coordinate_index.hpp"
#include "ngraph/coordinate_transform.hpp"

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
                if (axis < 0)
                {
                    axis += data_shape.size();
                }
                if (axis < 0 || axis >= data_shape.size())
                {
                    throw std::domain_error{
                        "axis for GatherElements exceeds allowed range [0, data_rank)"};
                }

                // in 1D case results can be achieved without additional calculations
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

                /*
                 assume data and indices are 6D and axis = 2
                 size of indices(N0,N1,N2,N3,N4,N5)
                 size of data (N0,N1,N2',N3,N4,N5)

                 the offset for indices will be
                 N5*N4*N3*N2*N1*n0 + N5*N4*N3*N2*n1 + N5*N4*N3*n2 + N5*N4*n3 + N5*n4 + n5
                 and for data
                 N5*N4*N3*N2'*N1*n0 + N5*N4*N3*N2'*n1 + N5*N4*N3*n2' + N5*N4*n3 + N5*n4 + n5
                 all values (except n2') are fixed or gradually increase
                 most of offset calculations are shared. We can rewrite offset for data as follows

                 N5*N4*N3*N2'*(N1*n0 + n1) + N5*N4*N3*n2' + (N5*N4*n3 + N5*n4 + n5)
                 N5*N4*N3*N2' - data_coeff
                 (N1*n0 + n1) - outer_sum
                 (N5*N4*n3 + N5*n4 + n5) - inner_sum
                */

                size_t outer_sum = 0, inner_sum = 0;

                // in 6D case with axis = 2
                // N5*N4*N3
                size_t max_inner_sum = ngraph::shape_size(
                    std::vector<size_t>(indices_shape.begin() + axis + 1, indices_shape.end()));
                // in 6D case with axis = 2
                // N5*N4*N3*N2'
                size_t data_coeff = ngraph::shape_size(
                    std::vector<size_t>(data_shape.begin() + axis, data_shape.end()));
                // in 6D case with axis = 2
                // N5*N4*N3*N2
                size_t outer_threshold_inc = ngraph::shape_size(
                    std::vector<size_t>(indices_shape.begin() + axis, indices_shape.end()));
                size_t outer_threshold = outer_threshold_inc;

                size_t count = ngraph::shape_size(indices_shape);
                int64_t data_count = ngraph::shape_size(data_shape);
                int64_t data_idx; // signed since indices is int32 or int64
                for (size_t i = 0; i < count;)
                {
                    data_idx = data_coeff * outer_sum + max_inner_sum * indices[i] + inner_sum;

                    if (data_idx < 0 || data_idx > data_count)
                    {
                        throw std::domain_error{"indices values of GatherElement exceed data size"};
                    }
                    out[i] = data[data_idx];

                    i++;
                    if (i == outer_threshold)
                    {
                        outer_sum++;
                        outer_threshold += outer_threshold_inc;
                    }

                    inner_sum++;
                    if (inner_sum == max_inner_sum)
                    {
                        inner_sum = 0;
                    }
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
