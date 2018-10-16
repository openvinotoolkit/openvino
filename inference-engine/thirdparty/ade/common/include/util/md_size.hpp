// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef UTIL_MD_SIZE_HPP
#define UTIL_MD_SIZE_HPP

#include <algorithm>
#include <array>
#include <initializer_list>

#include "util/assert.hpp"
#include "util/iota_range.hpp"
#include "util/checked_cast.hpp"

namespace util
{

/// Dinamically sized arbitrary dimensional size
template <std::size_t MaxDimensions>
struct DynMdSize final
{
    using SizeT = int;
    std::array<SizeT, MaxDimensions> sizes;
    std::size_t dims_cnt = 0;

    DynMdSize() = default;

    DynMdSize(std::initializer_list<SizeT> d):
        dims_cnt(util::checked_cast<decltype(this->dims_cnt)>(d.size()))
    {
        ASSERT(d.size() <= MaxDimensions);
        std::copy(d.begin(), d.end(), sizes.begin());
    }

    DynMdSize(const DynMdSize&) = default;
    DynMdSize& operator=(const DynMdSize&) = default;

    bool operator==(const DynMdSize& other) const
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

    bool operator!=(const DynMdSize& other) const
    {
        return !(*this == other);
    }

    SizeT& operator[](std::size_t index)
    {
        ASSERT(index < dims_count());
        return sizes[index];
    }

    const SizeT& operator[](std::size_t index) const
    {
        ASSERT(index < dims_count());
        return sizes[index];
    }

    SizeT* data()
    {
        return sizes.data();
    }

    const SizeT* data() const
    {
        return sizes.data();
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

    auto begin()
    ->decltype(this->sizes.begin())
    {
        return sizes.begin();
    }

    auto end()
    ->decltype(this->sizes.begin() + this->dims_count())
    {
        return sizes.begin() + dims_count();
    }

    auto begin() const
    ->decltype(this->sizes.begin())
    {
        return sizes.begin();
    }

    auto end() const
    ->decltype(this->sizes.begin() + this->dims_count())
    {
        return sizes.begin() + dims_count();
    }
};
}

#endif // UTIL_MD_SIZE_HPP
