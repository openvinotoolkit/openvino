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
    /// \brief A vector of axes.
    class AxisVector : public std::vector<size_t>
    {
    public:
        NGRAPH_API AxisVector(const std::initializer_list<size_t>& axes);

        NGRAPH_API AxisVector(const std::vector<size_t>& axes);

        NGRAPH_API AxisVector(const AxisVector& axes);

        NGRAPH_API explicit AxisVector(size_t n);

        template <class InputIterator>
        AxisVector(InputIterator first, InputIterator last)
            : std::vector<size_t>(first, last)
        {
        }

        NGRAPH_API AxisVector();

        NGRAPH_API ~AxisVector();

        NGRAPH_API AxisVector& operator=(const AxisVector& v);

        NGRAPH_API AxisVector& operator=(AxisVector&& v) noexcept;
    };

    template <>
    class NGRAPH_API AttributeAdapter<AxisVector>
        : public IndirectVectorValueAccessor<AxisVector, std::vector<int64_t>>
    {
    public:
        AttributeAdapter(AxisVector& value)
            : IndirectVectorValueAccessor<AxisVector, std::vector<int64_t>>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<AxisVector>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };

    NGRAPH_API
    std::ostream& operator<<(std::ostream& s, const AxisVector& axis_vector);
} // namespace ngraph
