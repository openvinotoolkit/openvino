// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef UTIL_RANGE_ITERATOR_HPP
#define UTIL_RANGE_ITERATOR_HPP

#include <iterator>
#include <type_traits>

#include "util/assert.hpp"

namespace util
{

inline namespace Range
{

template<typename R>
struct IterableRange
{
    // This iterator suitable only for range for, handle with care
    struct iterator
    {
        R range;
        bool end /*= false*/; // Need C++14

        using value_type = typename std::remove_reference<decltype(range.front())>::type;
        using pointer = value_type*;
        using reference = value_type&;
        using iterator_category = std::input_iterator_tag;
        using difference_type = std::ptrdiff_t;

        bool operator==(iterator const& other) const
        {
            if((range.empty() && other.end) ||
               (end           && other.range.empty()))
            {
                return true;
            }
            return false;
        }

        bool operator!=(iterator const& other) const
        {
            return !(*this == other);
        }

        auto operator*() -> decltype(range.front())
        {
            ASSERT(!range.empty());
            ASSERT(!end);
            return range.front();
        }

        auto operator*() const -> decltype(range.front())
        {
            ASSERT(!range.empty());
            ASSERT(!end);
            return range.front();
        }

        iterator& operator++()
        {
            ASSERT(!range.empty());
            ASSERT(!end);
            range.popFront();
            return *this;
        }
    };

    iterator begin()
    {
        auto& src = *static_cast<R*>(this);
        return iterator{src, false};
    }

    iterator end()
    {
        auto& src = *static_cast<R*>(this);
        return iterator{src, true};
    }
};

}

}
#endif // UTIL_RANGE_ITERATOR_HPP
