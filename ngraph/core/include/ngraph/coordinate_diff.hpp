// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <ostream>
#include <vector>

#include "ngraph/attribute_adapter.hpp"
#include "ngraph/ngraph_visibility.hpp"

namespace ngraph
{
    /// \brief A difference (signed) of tensor element coordinates.
    class CoordinateDiff : public std::vector<std::ptrdiff_t>
    {
    public:
        NGRAPH_API CoordinateDiff(const std::initializer_list<std::ptrdiff_t>& diffs);

        NGRAPH_API CoordinateDiff(const std::vector<std::ptrdiff_t>& diffs);

        NGRAPH_API CoordinateDiff(const CoordinateDiff& diffs);

        NGRAPH_API explicit CoordinateDiff(size_t n, std::ptrdiff_t initial_value = 0);

        template <class InputIterator>
        CoordinateDiff(InputIterator first, InputIterator last)
            : std::vector<std::ptrdiff_t>(first, last)
        {
        }

        NGRAPH_API ~CoordinateDiff();

        NGRAPH_API CoordinateDiff();

        NGRAPH_API CoordinateDiff& operator=(const CoordinateDiff& v);

        NGRAPH_API CoordinateDiff& operator=(CoordinateDiff&& v) noexcept;
    };

    template <>
    class NGRAPH_API AttributeAdapter<CoordinateDiff>
        : public IndirectVectorValueAccessor<CoordinateDiff, std::vector<int64_t>>

    {
    public:
        AttributeAdapter(CoordinateDiff& value)
            : IndirectVectorValueAccessor<CoordinateDiff, std::vector<int64_t>>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<CoordinateDiff>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };

    NGRAPH_API
    std::ostream& operator<<(std::ostream& s, const CoordinateDiff& coordinate_diff);
} // namespace ngraph
