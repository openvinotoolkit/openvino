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

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void normalize_l2(const T* data,
                              T* out,
                              const Shape& data_shape,
                              const AxisSet& reduction_axes,
                              float eps,
                              op::EpsMode eps_mode)
            {
                AxisSet axes = reduction_axes;
                if (reduction_axes.empty())
                {
                    std::vector<size_t> axes_vec(data_shape.size());
                    std::iota(axes_vec.begin(), axes_vec.end(), 0);
                    axes = AxisSet(axes_vec);
                }
                std::vector<T> sqr_data(shape_size(data_shape));
                for (size_t i = 0; i < shape_size(data_shape); i++)
                {
                    sqr_data[i] = data[i] * data[i];
                }

                Shape reduce_shape = data_shape;
                for (auto axis : axes)
                {
                    reduce_shape[axis] = 1;
                }

                std::vector<T> sum_data(shape_size(reduce_shape));
                sum(sqr_data.data(), sum_data.data(), data_shape, axes, true);
                autobroadcast_binop(data,
                                    sum_data.data(),
                                    out,
                                    data_shape,
                                    reduce_shape,
                                    op::AutoBroadcastSpec(op::AutoBroadcastType::NUMPY),
                                    [&eps, &eps_mode](T x, T y) -> T {
                                        T arg = (eps_mode == op::EpsMode::ADD)
                                                    ? y + eps
                                                    : std::max(y, static_cast<T>(eps));
                                        return x / std::sqrt(arg);
                                    });
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
