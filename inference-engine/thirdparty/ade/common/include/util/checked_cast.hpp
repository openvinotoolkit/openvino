// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef UTIL_CHECKED_CAST_HPP
#define UTIL_CHECKED_CAST_HPP

#include <limits>
#include <type_traits>

#include "util/assert.hpp"

#undef min
#undef max

namespace util
{

template <typename I, typename J, typename Handler>
inline typename std::enable_if< std::is_same<I, J>::value, I >::type
checked_cast_impl(J value, Handler&& /*handler*/)
{
    return value;
}

template <typename I, typename J, typename Handler>
inline typename std::enable_if<
   std::is_integral<I>::value && std::is_integral<J>::value &&
   std::is_signed<I>::value   && std::is_signed<J>::value &&
   !std::is_same<I,J>::value,
   I
>::type checked_cast_impl(J value, Handler&& handler)
{
    handler(value >= std::numeric_limits<I>::lowest() && value <= std::numeric_limits<I>::max(), value);
    return static_cast<I>(value);
}

template <typename I, typename J, typename Handler>
inline typename std::enable_if<
   std::is_integral<I>::value && std::is_integral<J>::value &&
   std::is_signed<I>::value   && std::is_unsigned<J>::value,
   I
>::type checked_cast_impl(J value, Handler&& handler)
{
    handler(value <= static_cast<typename std::make_unsigned<I>::type>(std::numeric_limits<I>::max()), value);
    return static_cast<I>(value);
}

template <typename I, typename J, typename Handler>
inline typename std::enable_if<
   std::is_integral<I>::value && std::is_integral<J>::value &&
   std::is_unsigned<I>::value && std::is_signed<J>::value,
   I
>::type checked_cast_impl(J value, Handler&& handler)
{
    handler(value >= 0 && static_cast<typename std::make_unsigned<J>::type>(value) <= std::numeric_limits<I>::max(), value);
    return static_cast<I>(value);
}

template <typename I, typename J, typename Handler>
inline typename std::enable_if<
   std::is_integral<I>::value && std::is_integral<J>::value &&
   std::is_unsigned<I>::value && std::is_unsigned<J>::value &&
   !std::is_same<I,J>::value,
   I
>::type checked_cast_impl(J value, Handler&& handler)
{
    handler(value <= std::numeric_limits<I>::max(), value);
    return static_cast<I>(value);
}

template <typename I, typename J, typename Handler>
inline typename std::enable_if<
   std::is_integral<I>::value && std::is_floating_point<J>::value,
   I
>::type checked_cast_impl(J value, Handler&& handler)
{
//   This criterion may fail in the following corner cases: (1) if I=int32_t, J=float, and value=2^31 (2) if I=int64_t, J=float, and value=2^63 (3) ...etc...
//   For example, consider I=int and J=float and value=2^31 Due to rounding, (float)INT32_MAX equals 2^31, not 2^31-1 So checking if value <= INT32_MAX would incorrectly pass
//   Well, I am not sure if we should especially address this corner case If you think we should,
//   we may need especial floating-point constants for upper boundaries Note that this problem does not impact the lower boundaries, which are exactly -2^31, -2^63, etc
    handler(value <= static_cast<J>(std::numeric_limits<I>::max()) && value >= static_cast<J>(std::numeric_limits<I>::lowest()), value);
    return static_cast<I>(value);
}

template <typename I, typename J, typename Handler>
inline typename std::enable_if<
   std::is_same<float, I>::value && std::is_same<double, J>::value,
   I
>::type checked_cast_impl(J value, Handler&& handler)
{
    handler(static_cast<double>(static_cast<float>(value)) == value, value);
    return static_cast<I>(value);
}

struct CheckedCastDefHandler final
{
    template<typename T>
    void operator()(bool valid, T&& /*value*/) const
    {
        ASSERT(valid);
    }
};

template <typename I, typename J>
I checked_cast(J value)
{
    return checked_cast_impl<I>(value, CheckedCastDefHandler{});
}

} // util

#endif // UTIL_CHECKED_CAST_HPP

