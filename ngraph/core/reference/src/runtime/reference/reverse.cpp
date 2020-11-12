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

#include <cmath>
#include <cstring>
#include <iterator>

#include "ngraph/check.hpp"
#include "ngraph/coordinate_range.hpp"
#include "ngraph/runtime/reference/reverse.hpp"

using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            void reverse(const char* arg,
                         char* out,
                         const Shape& arg_shape,
                         const Shape& out_shape,
                         const AxisSet& reversed_axes,
                         size_t elem_size)
            {
                NGRAPH_CHECK(shape_size(arg_shape) == shape_size(out_shape));

                auto dst_mem = out;

                for (const auto& in_coord : coordinates::reverse(arg_shape, reversed_axes))
                {
                    const auto src_index = in_coord.index();
                    const auto src_mem = arg + src_index * elem_size;
                    std::memcpy(dst_mem, src_mem, elem_size);
                    std::advance(dst_mem, elem_size);
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
