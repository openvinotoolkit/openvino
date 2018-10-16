// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef UTIL_MD_IO_HPP
#define UTIL_MD_IO_HPP

#include <ostream>

#include "util/md_size.hpp"
#include "util/md_span.hpp"

// md_* stream operators

namespace util
{

inline std::ostream& operator<<(std::ostream& os, const Span& span)
{
    os << "{" << span.begin << ", " << span.end << "}";
    return os;
}

template <std::size_t MaxDimensions>
inline std::ostream& operator<<(std::ostream& os, const DynMdSize<MaxDimensions>& size)
{
    os << "{";
    for (auto i: util::iota(size.dims_count()))
    {
        // TODO: join range
        if (0 != i)
        {
            os << ", ";
        }
        os << size[i];
    }
    os << "}";
    return os;
}

template <std::size_t MaxDimensions>
inline std::ostream& operator<<(std::ostream& os, const DynMdSpan<MaxDimensions>& span)
{
    os << "{";
    for (auto i: util::iota(span.dims_count()))
    {
        // TODO: join range
        if (0 != i)
        {
            os << ", ";
        }
        os << span[i];
    }
    os << "}";
    return os;
}

}

#endif // UTIL_MD_IO_HPP
