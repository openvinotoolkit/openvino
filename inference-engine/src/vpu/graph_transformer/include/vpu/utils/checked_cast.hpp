// Copyright (C) 2018-2019 Intel Corporation
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

template <typename I, typename J>
typename std::enable_if<
        std::is_same<I, J>::value,
    I>::type checked_cast(J value) {
    return value;
}

template <typename I, typename J>
typename std::enable_if<
        std::is_integral<I>::value && std::is_integral<J>::value &&
        std::is_signed<I>::value && std::is_signed<J>::value &&
        !std::is_same<I, J>::value,
    I>::type checked_cast(J value) {
    IE_ASSERT(value >= std::numeric_limits<I>::lowest()) << value;
    IE_ASSERT(value <= std::numeric_limits<I>::max()) << value;
    return static_cast<I>(value);
}

template <typename I, typename J>
typename std::enable_if<
        std::is_integral<I>::value && std::is_integral<J>::value &&
        std::is_signed<I>::value && std::is_unsigned<J>::value,
    I>::type checked_cast(J value) {
    IE_ASSERT(value <= static_cast<typename std::make_unsigned<I>::type>(std::numeric_limits<I>::max())) << value;
    return static_cast<I>(value);
}

template <typename I, typename J>
typename std::enable_if<
        std::is_integral<I>::value && std::is_integral<J>::value &&
        std::is_unsigned<I>::value && std::is_signed<J>::value,
    I>::type checked_cast(J value) {
    IE_ASSERT(value >= 0) << value;
    // coverity[result_independent_of_operands]
    IE_ASSERT(static_cast<typename std::make_unsigned<J>::type>(value) <= std::numeric_limits<I>::max()) << value;
    return static_cast<I>(value);
}

template <typename I, typename J>
typename std::enable_if<
        std::is_integral<I>::value && std::is_integral<J>::value &&
        std::is_unsigned<I>::value && std::is_unsigned<J>::value &&
        !std::is_same<I, J>::value,
    I>::type checked_cast(J value) {
    // coverity[result_independent_of_operands]
    IE_ASSERT(value <= std::numeric_limits<I>::max()) << value;
    return static_cast<I>(value);
}

template <typename I, typename J>
typename std::enable_if<
        std::is_integral<I>::value && std::is_floating_point<J>::value,
    I>::type checked_cast(J value) {
    IE_ASSERT(value <= static_cast<J>(std::numeric_limits<I>::max())) << value;
    IE_ASSERT(value >= static_cast<J>(std::numeric_limits<I>::lowest())) << value;
    return static_cast<I>(value);
}

template <typename I, typename J>
typename std::enable_if<
        std::is_same<float, I>::value && std::is_same<double, J>::value,
    I>::type checked_cast(J value) {
    IE_ASSERT(static_cast<double>(static_cast<float>(value)) == value) << value;
    return static_cast<I>(value);
}

}  // namespace vpu
