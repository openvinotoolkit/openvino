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

#include <cstring>

#include "ngraph/check.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename DataType, typename IndicesType>
            void scatter_elem_update(const DataType* input_data,
                                     const IndicesType* indices,
                                     const DataType* updates,
                                     const int64_t& axis,
                                     DataType* out_buf,
                                     const Shape& data_shape,
                                     const Shape& indices_shape)
            {
                // Copy inputs to out
                std::memcpy(out_buf, input_data, sizeof(DataType) * shape_size(data_shape));

                // 3D example
                // output[indices[i][j][k]][j][k] = updates[i][j][k] if axis = 0,
                // output[i][indices[i][j][k]][k] = updates[i][j][k] if axis = 1,
                // output[i][j][indices[i][j][k]] = updates[i][j][k] if axis = 2

                CoordinateTransform indices_transform{indices_shape};
                CoordinateTransform data_transform{data_shape};

                for (const Coordinate& indices_cord : indices_transform)
                {
                    const size_t indices_idx = indices_transform.index(indices_cord);
                    Coordinate out_cord(indices_cord);
                    out_cord.at(axis) = indices[indices_idx];
                    NGRAPH_CHECK(data_transform.has_source_coordinate(out_cord),
                                 "Provided index coordinates are out of input data bounds: ",
                                 out_cord,
                                 ".");
                    out_buf[data_transform.index(out_cord)] = updates[indices_idx];
                }
            }
        }
    }
}
