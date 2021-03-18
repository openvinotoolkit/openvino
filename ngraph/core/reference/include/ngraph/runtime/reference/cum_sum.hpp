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
#include <map>
#include <utility>
#include <vector>

#include "ngraph/coordinate_transform.hpp"
#include "ngraph/type/bfloat16.hpp"
#include "ngraph/type/float16.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T, typename P>
            void cumsum(const T* arg,
                        const P* axis_tensor,
                        T* out,
                        const Shape& tensor_shape,
                        const bool exclusive,
                        const bool reverse)
            {
                CoordinateTransform temp_transform(tensor_shape);
                for (const Coordinate& output_coord : temp_transform)
                {
                    out[temp_transform.index(output_coord)] = 0;
                }

                P axis = axis_tensor[0];
                P rank = tensor_shape.size();

                if (axis < -rank || axis > rank)
                {
                    throw ngraph_error("axis must be in the range [-rank, rank]");
                }
                axis = axis < 0 ? rank + axis : axis;

                auto get_key = [&, axis](const Coordinate& coord) -> Coordinate {
                    Coordinate result(coord.size(), 0);
                    result[axis] = coord[axis];

                    for (size_t i = 0; i < coord.size(); i++)
                    {
                        result[i] = coord[i] - result[i];
                    }
                    return result;
                };

                auto update_output_buffer =
                    [&](size_t input_index,
                        size_t output_index,
                        T& prev,
                        std::vector<std::pair<size_t, T>>& tensor_vec) -> void {
                    tensor_vec[input_index].second = prev + tensor_vec[input_index].second;
                    out[tensor_vec[output_index].first] = tensor_vec[input_index].second;

                    // update prev to hold the last result value to compute ruuning sum for
                    // subsequent iter
                    prev = out[tensor_vec[output_index].first];
                };

                auto cum_sum =
                    [&, exclusive, reverse](std::vector<std::pair<size_t, T>>& tensor_vec) {
                        if (!reverse)
                        {
                            T prev = 0;
                            for (size_t i = 0; i < tensor_vec.size(); i++)
                            {
                                if (exclusive && i == 0)
                                {
                                    out[tensor_vec[i].first] = prev;
                                    continue;
                                }
                                // we will compute running sum of j-1 elements if exlusive=1 or else
                                // for j elements if exclusive = 0
                                size_t arg_index = exclusive == 1 ? i - 1 : i;
                                update_output_buffer(arg_index, i, prev, tensor_vec);
                            }
                        }
                        else // reverse == true
                        {
                            T prev = 0;
                            for (size_t i = tensor_vec.size(); i-- > 0;)
                            {
                                if (exclusive && i == tensor_vec.size() - 1)
                                {
                                    out[tensor_vec[i].first] = prev;
                                    continue;
                                }
                                // we will compute running sum of j-1 elements if exlusive=1 or else
                                // for j elements if exclusive = 0
                                size_t arg_index = exclusive == 1 ? i + 1 : i;
                                update_output_buffer(arg_index, i, prev, tensor_vec);
                            }
                        }
                    };

                // Map to collect tensor elements belonging to the same axis
                std::map<Coordinate, std::vector<std::pair<size_t, T>>> map_cooord_to_val;
                CoordinateTransform input_transform(tensor_shape);
                for (const Coordinate& input_coord : input_transform)
                {
                    // points to the current element in the input tensor
                    T current = arg[input_transform.index(input_coord)];
                    auto key = get_key(input_coord);
                    auto index = input_transform.index(input_coord);
                    if (map_cooord_to_val.find(key) != map_cooord_to_val.end())
                    {
                        map_cooord_to_val[key].push_back(std::make_pair(index, current));
                    }
                    else
                    {
                        map_cooord_to_val.insert({key, std::vector<std::pair<size_t, T>>()});
                        map_cooord_to_val[key].push_back(std::make_pair(index, current));
                    }
                }
                // iterate the map and perform cumulative sum over the give axis
                for (auto& it : map_cooord_to_val)
                {
                    cum_sum(it.second);
                }
            }
        }
    }
}
