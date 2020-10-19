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

// TODO REMOVE INCLUDES
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/shape_util.hpp"

#include "tile.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void broadcast(const T* arg,
                           T* out,
                           const Shape& in_shape,
                           const Shape& out_shape,
                           const AxisSet& broadcast_axes)
            {
                std::cout << "in_shape: ";
                for (int i = 0; i < in_shape.size(); ++i)
                {
                    std::cout << in_shape[i] << ", ";
                }
                std::cout << "\n";

                std::cout << "out_shape: ";
                for (int i = 0; i < out_shape.size(); ++i)
                {
                    std::cout << out_shape[i] << ", ";
                }
                std::cout << "\n";

                std::cout << "broadcast_axes: ";
                for (auto& axis : broadcast_axes)
                {
                    std::cout << axis << ", ";
                }
                std::cout << "\n";

                // TODO CHECK
                Shape adjusted_in_shape = in_shape;
                for (const auto& axis : broadcast_axes)
                {
                    if (adjusted_in_shape.size() < out_shape.size())
                    {
                        adjusted_in_shape.insert(adjusted_in_shape.begin() + axis, 1);
                    }
                }
                // TODO assert rank
                std::vector<int64_t> repeats(out_shape.size());
                for (size_t i = 0; i < repeats.size(); ++i)
                {
                    // TODO CHECK IF OUT OF RANGE
                    repeats[i] = out_shape[i] / adjusted_in_shape[i]; // TODO CHECK ROUND
                }
                return tile(reinterpret_cast<const char*>(arg),
                            reinterpret_cast<char*>(out),
                            adjusted_in_shape,
                            out_shape,
                            sizeof(T),
                            repeats);
            }
        }
    }
}
