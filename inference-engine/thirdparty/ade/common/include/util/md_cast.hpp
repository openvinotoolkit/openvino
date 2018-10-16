// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef UTIL_MD_CAST_HPP
#define UTIL_MD_CAST_HPP

namespace util
{
// TODO: find a proper place for this
constexpr static const std::size_t MaxDimensions = 6;

namespace detail
{
template<typename Target>
struct md_cast_helper; // Undefined
}

template<typename Dst, typename Src>
Dst md_cast(const Src& src)
{
    return detail::md_cast_helper<Dst>(src);
}
}

#endif // UTIL_MD_CAST_HPP
