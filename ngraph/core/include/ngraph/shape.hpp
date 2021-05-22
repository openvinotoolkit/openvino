// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdio>
#include <vector>

#include "ngraph/attribute_adapter.hpp"
#include "ngraph/axis_set.hpp"
#include "ngraph/ngraph_visibility.hpp"
#include "ngraph/strides.hpp"

namespace ngraph
{
    /// \brief Shape for a tensor.
    class Shape : public std::vector<size_t>
    {
    public:
        NGRAPH_API Shape();

        NGRAPH_API Shape(const std::initializer_list<size_t>& axis_lengths);

        NGRAPH_API Shape(const std::vector<size_t>& axis_lengths);

        NGRAPH_API Shape(const Shape& axis_lengths);

        NGRAPH_API explicit Shape(size_t n, size_t initial_value = 0);

        NGRAPH_API ~Shape();

        template <class InputIterator>
        Shape(InputIterator first, InputIterator last)
            : std::vector<size_t>(first, last)
        {
        }

        NGRAPH_API Shape& operator=(const Shape& v);
        NGRAPH_API Shape& operator=(Shape&& v) noexcept;
    };

    template <>
    class NGRAPH_API AttributeAdapter<Shape>
        : public IndirectVectorValueAccessor<Shape, std::vector<int64_t>>

    {
    public:
        AttributeAdapter(Shape& value)
            : IndirectVectorValueAccessor<Shape, std::vector<int64_t>>(value)
        {
        }
        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<Shape>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };

    /// Number of elements in spanned by a shape
    template <typename SHAPE_TYPE>
    size_t shape_size(const SHAPE_TYPE& shape)
    {
        size_t size = 1;
        for (auto d : shape)
        {
            size *= d;
        }
        return size;
    }

    /// Row-major strides for a shape
    template <typename SHAPE_TYPE>
    std::vector<size_t> row_major_strides(const SHAPE_TYPE& shape)
    {
        std::vector<size_t> strides(shape.size());
        size_t s = 1;
        auto st = strides.rbegin();
        for (auto d = shape.rbegin(); d != shape.rend() && st != strides.rend(); d++, st++)
        {
            *st = s;
            s *= *d;
        }
        return strides;
    }

    template <typename SHAPE_TYPE>
    size_t row_major_stride(const SHAPE_TYPE& shape, size_t axis)
    {
        size_t s = 1;
        for (size_t i = shape.size(); i-- > axis + 1;)
        {
            s *= shape[i];
        }
        return s;
    }

    template <typename SHAPE_TYPE>
    inline bool is_scalar(const SHAPE_TYPE& shape)
    {
        return 0 == shape.size();
    }

    template <typename SHAPE_TYPE>
    inline bool is_vector(const SHAPE_TYPE& shape)
    {
        return 1 == shape.size();
    }

    NGRAPH_API
    std::ostream& operator<<(std::ostream& s, const Shape& shape);
} // namespace ngraph
