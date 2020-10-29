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

#include "ngraph/runtime/reference/broadcast.hpp"
#include "ngraph/runtime/reference/tile.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            void broadcast(const char* arg,
                           char* out,
                           const Shape& in_shape,
                           const Shape& out_shape,
                           const AxisSet& broadcast_axes,
                           size_t elem_size)
            {
                const auto output_rank = std::max(in_shape.size(), out_shape.size());
                Shape adjusted_in_shape = in_shape;
                for (const auto& axis : broadcast_axes)
                {
                    if (adjusted_in_shape.size() < output_rank)
                    {
                        adjusted_in_shape.insert(adjusted_in_shape.begin() + axis, 1);
                    }
                }
                Shape adjusted_out_shape = out_shape;
                adjusted_out_shape.insert(
                    adjusted_out_shape.begin(), output_rank - adjusted_out_shape.size(), 1);
                std::vector<int64_t> repeats(output_rank);
                for (size_t i = 0; i < repeats.size(); ++i)
                {
                    repeats[i] = adjusted_out_shape[i] / adjusted_in_shape[i];
                }

                return tile(arg, out, adjusted_in_shape, adjusted_out_shape, elem_size, repeats);
            }
        }
    }
}