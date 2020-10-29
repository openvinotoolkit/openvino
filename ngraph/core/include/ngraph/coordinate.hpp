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
#include <vector>

#include "ngraph/attribute_adapter.hpp"
#include "ngraph/axis_set.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    /// \brief Coordinates for a tensor element
    class Coordinate : public std::vector<size_t>
    {
    public:
        NGRAPH_API Coordinate();
        NGRAPH_API Coordinate(const std::initializer_list<size_t>& axes);

        NGRAPH_API Coordinate(const Shape& shape);

        NGRAPH_API Coordinate(const std::vector<size_t>& axes);

        NGRAPH_API Coordinate(const Coordinate& axes);

        NGRAPH_API Coordinate(size_t n, size_t initial_value = 0);

        NGRAPH_API ~Coordinate();

        template <class InputIterator>
        Coordinate(InputIterator first, InputIterator last)
            : std::vector<size_t>(first, last)
        {
        }

        NGRAPH_API Coordinate& operator=(const Coordinate& v);

        NGRAPH_API Coordinate& operator=(Coordinate&& v) noexcept;
    };

    template <>
    class NGRAPH_API AttributeAdapter<Coordinate>
        : public IndirectVectorValueAccessor<Coordinate, std::vector<int64_t>>
    {
    public:
        AttributeAdapter(Coordinate& value)
            : IndirectVectorValueAccessor<Coordinate, std::vector<int64_t>>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<Coordinate>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };

    NGRAPH_API
    std::ostream& operator<<(std::ostream& s, const Coordinate& coordinate);
}
