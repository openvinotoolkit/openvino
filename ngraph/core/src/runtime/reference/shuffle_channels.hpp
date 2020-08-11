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
#include <numeric>

#include "ngraph/runtime/reference/reshape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void shuffle_channels(const T* arg, T* out, const Shape &data_shape, int64_t axis, int64_t group)
            {
                const Shape& ds = data_shape;

                // in general the resulting shape should contain the following values:
                // [0]: ds[0] * ds[1] * ... * ds[m_axis-1] (or 1 if m_axis == 0)
                // [1]: m_group
                // [2]: ds[axis] / m_group
                // [3]: ds[axis+1] * ds[axis+2] * ... * ds[ds.size()-1] (or 1 if m_axis points to the last elem
                //                                                       of ds)
                Shape pre_reshape_shape(4, 1);

                size_t axis_zb = axis >= 0 ? axis : axis + data_shape.size();
                for (size_t i = 0; i < axis_zb; ++i) {
                    pre_reshape_shape[0] *= ds[i];
                }

                pre_reshape_shape[1] = group;
                pre_reshape_shape[2] = ds[axis_zb] / group;

                for (size_t i = axis_zb + 1; i < ds.size(); ++i) {
                    pre_reshape_shape[3] *= ds[i];
                }
                AxisVector axes_order(data_shape.size());
                std::iota(axes_order.begin(), axes_order.end(), 0);

                std::vector<T> reshaped(shape_size(data_shape));
                reshape(arg, reshaped.data(), data_shape, axes_order, pre_reshape_shape);

                Shape transpose_axes_order = {0, 2, 1, 3};
                Shape transposed_shape = pre_reshape_shape;

                for (size_t i = 0; i < transpose_axes_order.size(); ++i) {
                    transposed_shape[i] = data_shape.at(transpose_axes_order.at(i));
                }
                auto axis_vector = AxisVector{begin(transpose_axes_order), end(transpose_axes_order)};
                std::vector<T> transposed(shape_size(data_shape));
                reshape(reshaped.data(), transposed.data(), pre_reshape_shape, axis_vector, transposed_shape);

                reshape(transposed.data(), out, transposed_shape, axes_order, data_shape);
            }
        }
    }
}
