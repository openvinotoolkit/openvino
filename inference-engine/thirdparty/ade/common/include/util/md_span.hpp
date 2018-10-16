// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef UTIL_MD_SPAN_HPP
#define UTIL_MD_SPAN_HPP

#include <algorithm>
#include <array>
#include <initializer_list>

#include "util/assert.hpp"
#include "util/iota_range.hpp"
#include "util/md_size.hpp"
#include "util/checked_cast.hpp"

namespace util {

struct Span final
{
    int begin = 0;
    int end   = 0;

    Span() = default;

    Span(int b, int e) : begin(b), end(e)
    {
        ASSERT(b <= e);
    }

    int length() const
    {
        return end - begin;
    }

    bool operator==(const Span& other) const
    {
        return begin == other.begin &&
               end   == other.end;
    }

    bool operator!=(const Span& other) const
    {
        return !(*this == other);
    }

    bool intersectsWith(const Span& other) const
    {
        auto s = std::max(begin, other.begin);
        auto e = std::min(end, other.end);
        return (s < e);
    }
};

/// Dinamically sized arbitrary dimensional span
template <size_t MaxDimensions>
struct DynMdSpan final
{
    std::array<Span, MaxDimensions> spans;
    std::size_t dims_cnt = 0;

    DynMdSpan() = default;

    DynMdSpan(std::initializer_list<Span> d):
        dims_cnt(util::checked_cast<decltype(this->dims_cnt)>(d.size()))
    {
        ASSERT(d.size() <= MaxDimensions);
        std::copy(d.begin(), d.end(), spans.begin());
    }

    DynMdSpan(const DynMdSpan&) = default;
    DynMdSpan& operator=(const DynMdSpan&) = default;

    bool operator==(const DynMdSpan& other) const
    {
        if (dims_count() != other.dims_count())
        {
            return false;
        }

        for (auto i: util::iota(dims_count()))
        {
            if ((*this)[i] != other[i])
            {
                return false;
            }
        }
        return true;
    }

    bool operator!=(const DynMdSpan& other) const
    {
        return !(*this == other);
    }

    Span& operator[](std::size_t index)
    {
        ASSERT(index < dims_count());
        return spans[index];
    }

    const Span& operator[](std::size_t index) const
    {
        ASSERT(index < dims_count());
        return spans[index];
    }

    Span* data()
    {
        return spans.data();
    }

    const Span* data() const
    {
        return spans.data();
    }

    std::size_t dims_count() const
    {
        return dims_cnt;
    }

    void redim(std::size_t count)
    {
        ASSERT(count <= MaxDimensions);
        dims_cnt = count;
    }

    auto begin() -> decltype(spans.begin())
    {
        return spans.begin();
    }

    auto end() -> decltype(spans.begin())
    {
        return spans.begin() + dims_count();
    }

    auto begin() const -> decltype(spans.begin())
    {
        return spans.begin();
    }

    auto end() const -> decltype(spans.begin())
    {
        return spans.begin() + dims_count();
    }

    auto cbegin() const -> decltype(spans.cbegin())
    {
        return spans.cbegin();
    }

    auto cend() const -> decltype(spans.cbegin())
    {
        return spans.cbegin() + dims_count();
    }

    DynMdSize<MaxDimensions> size() const
    {
        DynMdSize<MaxDimensions> ret;
        ret.redim(dims_count());
        std::transform(begin(), end(), ret.begin(), [](const Span& s)
        {
            return s.length();
        });
        return ret;
    }

    DynMdSize<MaxDimensions> origin() const
    {
        DynMdSize<MaxDimensions> ret;
        ret.redim(dims_count());
        std::transform(begin(), end(), ret.begin(), [](const Span& s)
        {
            return s.begin;
        });
        return ret;
    }
};

template<size_t MaxDimensions>
DynMdSpan<MaxDimensions> make_span(const DynMdSize<MaxDimensions>& origin,
                                   const DynMdSize<MaxDimensions>& size)
{
    ASSERT(origin.dims_count() == size.dims_count());
    const auto dims_count = origin.dims_count();
    DynMdSpan<MaxDimensions> ret;
    ret.redim(dims_count);
    for (auto i: util::iota(dims_count))
    {
        ret[i] = Span(origin[i], origin[i] + size[i]);
    }
    return ret;
}

template <size_t MaxDimensions1, size_t MaxDimensions2>
bool spanIntersects(const DynMdSpan<MaxDimensions1>& span1, const DynMdSpan<MaxDimensions2>& span2)
{
    if (span1.dims_count() != span2.dims_count())
        return false;

    for (auto i : util::iota(span1.dims_count()))
    {
        if (!span1[i].intersectsWith(span2[i]))
            return false;
    }

    return true;
}

template<size_t MaxDimensions>
inline DynMdSpan<MaxDimensions> operator+(const DynMdSpan<MaxDimensions>& s1, const DynMdSize<MaxDimensions>& s2)
{
    ASSERT(s1.dims_count() == s2.dims_count());
    DynMdSpan<MaxDimensions> ret;
    ret.redim(s1.dims_count());
    for (auto i: util::iota(ret.dims_count()))
    {
        ret[i].begin = s1[i].begin + s2[i];
        ret[i].end   = s1[i].end   + s2[i];
    }
    return ret;
}

template<size_t MaxDimensions>
inline DynMdSpan<MaxDimensions> operator+(const DynMdSize<MaxDimensions>& s1, const DynMdSpan<MaxDimensions>& s2)
{
    return s2 + s1;
}

template<size_t MaxDimensions>
inline DynMdSpan<MaxDimensions> operator-(const DynMdSpan<MaxDimensions>& s1, const DynMdSize<MaxDimensions>& s2)
{
    ASSERT(s1.dims_count() == s2.dims_count());
    DynMdSpan<MaxDimensions> ret;
    ret.redim(s1.dims_count());
    for (auto i: util::iota(ret.dims_count()))
    {
        ret[i].begin = s1[i].begin - s2[i];
        ret[i].end   = s1[i].end   - s2[i];
    }
    return ret;
}

}

#endif // UTIL_MD_SPAN_HPP
