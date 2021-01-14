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

#include "ngraph/coordinate_index.hpp"

#include "ngraph/coordinate.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    std::size_t coordinate_index(const Coordinate& c, const Shape& s)
    {
        if (c.size() < s.size())
        {
            throw std::domain_error("Coordinate rank is less than shape rank.");
        }
        std::size_t index = 0;
        std::size_t stride = 1;
        std::size_t const padding = c.size() - s.size();

        for (std::size_t axis = s.size(); axis-- > 0;)
        {
            if (s[axis] > 1)
            {
                index += c[axis + padding] * stride;
                stride *= s[axis];
            }
        }

        return index;
    }
} // namespace ngraph
