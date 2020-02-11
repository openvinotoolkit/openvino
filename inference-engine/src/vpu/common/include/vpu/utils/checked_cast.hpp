// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifdef __INTEL_COMPILER
#pragma warning disable: 54
#endif

#include <type_traits>
#include <limits>

#include <details/ie_exception.hpp>

namespace vpu {

namespace details {

template <bool Cond, class Func>
inline typename std::enable_if<Cond, void>::type runIf(Func&& func) {
    func();
}
template <bool Cond, class Func>
inline typename std::enable_if<!Cond, void>::type runIf(Func&&) {
}

// To overcame syntax parse error, when `>` comparison operator is threated as template closing bracket
template <typename T1, typename T2>
constexpr inline bool Greater(T1&& v1, T2&& v2) {
    return v1 > v2;
}

}  // namespace details

template <typename OutT, typename InT>
inline typename std::enable_if<
        std::is_same<OutT, InT>::value,
OutT>::type checked_cast(InT value) {
    return value;
}

template <typename OutT, typename InT>
inline typename std::enable_if<
        std::is_integral<OutT>::value && std::is_integral<InT>::value &&
        std::is_signed<OutT>::value && std::is_signed<InT>::value &&
        !std::is_same<OutT, InT>::value,
OutT>::type checked_cast(InT value) {
    details::runIf<
        std::numeric_limits<InT>::lowest() < std::numeric_limits<OutT>::lowest()
    >([&] {
        IE_ASSERT(value >= std::numeric_limits<OutT>::lowest()) << value;
    });
    details::runIf<
        details::Greater(std::numeric_limits<InT>::max(), std::numeric_limits<OutT>::max())
    >([&] {
        IE_ASSERT(value <= std::numeric_limits<OutT>::max()) << value;
    });

    return static_cast<OutT>(value);
}

template <typename OutT, typename InT>
typename std::enable_if<
        std::is_integral<OutT>::value && std::is_integral<InT>::value &&
        std::is_unsigned<OutT>::value && std::is_unsigned<InT>::value &&
        !std::is_same<OutT, InT>::value,
    OutT>::type checked_cast(InT value) {
    details::runIf<
        details::Greater(std::numeric_limits<InT>::max(), std::numeric_limits<OutT>::max())
    >([&] {
        IE_ASSERT(value <= std::numeric_limits<OutT>::max()) << value;
    });

    return static_cast<OutT>(value);
}

template <typename OutT, typename InT>
typename std::enable_if<
        std::is_integral<OutT>::value && std::is_integral<InT>::value &&
        std::is_signed<OutT>::value && std::is_unsigned<InT>::value,
    OutT>::type checked_cast(InT value) {
    details::runIf<
        details::Greater(std::numeric_limits<InT>::max(), static_cast<typename std::make_unsigned<OutT>::type>(std::numeric_limits<OutT>::max()))
    >([&] {
        IE_ASSERT(value <= static_cast<typename std::make_unsigned<OutT>::type>(std::numeric_limits<OutT>::max())) << value;
    });

    return static_cast<OutT>(value);
}

template <typename OutT, typename InT>
typename std::enable_if<
        std::is_integral<OutT>::value && std::is_integral<InT>::value &&
        std::is_unsigned<OutT>::value && std::is_signed<InT>::value,
    OutT>::type checked_cast(InT value) {
    IE_ASSERT(value >= 0) << value;
    details::runIf<
        details::Greater(static_cast<typename std::make_unsigned<InT>::type>(std::numeric_limits<InT>::max()), std::numeric_limits<OutT>::max())
    >([&] {
        IE_ASSERT(static_cast<typename std::make_unsigned<InT>::type>(value) <= std::numeric_limits<OutT>::max()) << value;
    });

    return static_cast<OutT>(value);
}

template <typename OutT, typename InT>
typename std::enable_if<
        std::is_integral<OutT>::value && std::is_floating_point<InT>::value,
    OutT>::type checked_cast(InT value) {
    IE_ASSERT(value <= static_cast<InT>(std::numeric_limits<OutT>::max())) << value;
    IE_ASSERT(value >= static_cast<InT>(std::numeric_limits<OutT>::lowest())) << value;

    return static_cast<OutT>(value);
}

template <typename OutT, typename InT>
typename std::enable_if<
        std::is_same<float, OutT>::value && std::is_same<double, InT>::value,
    OutT>::type checked_cast(InT value) {
    IE_ASSERT(static_cast<double>(static_cast<float>(value)) == value) << value;

    return static_cast<OutT>(value);
}

}  // namespace vpu
