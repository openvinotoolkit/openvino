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
            template <typename T>
            void sum_region_across_axes(const T* arg,
                                        size_t current_axis_index,
                                        const std::vector<size_t>& axes,
                                        Coordinate& sum_coord,
                                        T& square_sum,
                                        const std::vector<size_t>& begin_area,
                                        const std::vector<size_t>& end_area,
                                        const CoordinateTransform& input_transform)
            {
                // all nested axes were visited
                if (current_axis_index == axes.size())
                {
                    square_sum += arg[input_transform.index(sum_coord)] *
                                  arg[input_transform.index(sum_coord)];
                    return;
                }
                auto current_axis = axes[current_axis_index];
                for (auto current_axis_coord = begin_area[current_axis];
                     current_axis_coord < end_area[current_axis];
                     ++current_axis_coord)
                {
                    sum_coord.at(current_axis) = current_axis_coord;
                    sum_region_across_axes(arg,
                                           current_axis_index + 1,
                                           axes,
                                           sum_coord,
                                           square_sum,
                                           begin_area,
                                           end_area,
                                           input_transform);
                }
            }

            template <typename T>
            void lrn(const T* arg,
                     const AxisSet& axes,
                     T* out,
                     const Shape& arg_shape,
                     double dalpha,
                     double dbeta,
                     double dbias,
                     size_t size)
            {
                T alpha = static_cast<T>(dalpha);
                T beta = static_cast<T>(dbeta);
                T bias = static_cast<T>(dbias);

                std::vector<size_t> begin_area(arg_shape.size());
                std::vector<size_t> end_area(arg_shape.size());

                CoordinateTransform input_transform(arg_shape);
                for (const Coordinate& in_coord : input_transform)
                {
                    // area determined by in_coord local neighborhood
                    for (const auto& axis_coord : axes)
                    {
                        begin_area[axis_coord] =
                            std::max<int>(0, in_coord.at(axis_coord) - (size - 1) / 2);
                        end_area[axis_coord] = std::min<int>(
                            arg_shape.at(axis_coord), in_coord.at(axis_coord) + (size - 1) / 2 + 1);
                    }

                    T square_sum = 0;
                    auto sum_coord = in_coord;
                    auto axes_vec = std::vector<size_t>(axes.begin(), axes.end());
                    sum_region_across_axes(arg,
                                           0,
                                           axes_vec,
                                           sum_coord,
                                           square_sum,
                                           begin_area,
                                           end_area,
                                           input_transform);

                    T x = arg[input_transform.index(in_coord)];
                    out[input_transform.index(in_coord)] =
                        x / (std::pow(bias + (alpha / size) * square_sum, beta));
                }
            }
        }
    }
}
