// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
} // namespace ngraph
