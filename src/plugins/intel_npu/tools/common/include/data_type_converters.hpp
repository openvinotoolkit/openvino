//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/element_type_traits.hpp"

#include <limits>

namespace npu {
namespace utils {

namespace details {

template <bool Cond, class Func>
std::enable_if_t<Cond> staticIf(Func&& func) {
    func();
}

template <bool Cond, class Func>
std::enable_if_t<!Cond> staticIf(Func&&) {
}

// To overcome the syntax parse error, when `>` comparison operator is treated as
// template closing bracket
template <typename T1, typename T2>
constexpr bool Greater(T1&& v1, T2&& v2) {
    return v1 > v2;
}

}  // namespace details

//
// Bool logic
//

template <typename T>
using not_ = std::negation<T>;

template <typename... Ts>
using or_ = std::disjunction<Ts...>;

template <typename... Ts>
using and_ = std::conjunction<Ts...>;

//
// enable_if
//

template <typename T, typename... Args>
using enable_t = std::enable_if_t<(Args::value && ...), T>;

//
// Standart data types
//

template <typename OutT, typename InT>
enable_t<OutT, std::is_same<OutT, InT>> checked_cast(InT value) {
    return value;
}

template <typename OutT, typename InT>
enable_t<OutT, std::is_integral<InT>, std::is_signed<InT>, std::is_integral<OutT>, std::is_signed<OutT>,
         not_<std::is_same<OutT, InT>>>
checked_cast(InT value) {
    details::staticIf<std::numeric_limits<InT>::lowest() < std::numeric_limits<OutT>::lowest()>([&] {
        OPENVINO_ASSERT(value >= std::numeric_limits<OutT>::lowest(), "Can not safely cast ",
                        static_cast<int64_t>(value), " from ", ov::element::from<InT>(), " to ",
                        ov::element::from<OutT>());
    });

    details::staticIf<details::Greater(std::numeric_limits<InT>::max(), std::numeric_limits<OutT>::max())>([&] {
        OPENVINO_ASSERT(value <= std::numeric_limits<OutT>::max(), "Can not safely cast ", static_cast<int64_t>(value),
                        " from ", ov::element::from<InT>(), " to ", ov::element::from<OutT>());
    });

    return static_cast<OutT>(value);
}

template <typename OutT, typename InT>
enable_t<OutT, std::is_integral<InT>, std::is_unsigned<InT>, std::is_integral<OutT>, std::is_unsigned<OutT>,
         not_<std::is_same<OutT, InT>>>
checked_cast(InT value) {
    details::staticIf<details::Greater(std::numeric_limits<InT>::max(), std::numeric_limits<OutT>::max())>([&] {
        OPENVINO_ASSERT(value <= std::numeric_limits<OutT>::max(), "Can not safely cast ", static_cast<uint64_t>(value),
                        " from ", ov::element::from<InT>(), " to ", ov::element::from<OutT>());
    });

    return static_cast<OutT>(value);
}

template <typename OutT, typename InT>
enable_t<OutT, std::is_integral<InT>, std::is_unsigned<InT>, std::is_integral<OutT>, std::is_signed<OutT>> checked_cast(
        InT value) {
    details::staticIf<details::Greater(std::numeric_limits<InT>::max(),
                                       static_cast<std::make_unsigned_t<OutT>>(std::numeric_limits<OutT>::max()))>([&] {
        OPENVINO_ASSERT(value <= static_cast<std::make_unsigned_t<OutT>>(std::numeric_limits<OutT>::max()),
                        "Can not safely cast ", static_cast<uint64_t>(value), " from ", ov::element::from<InT>(),
                        " to ", ov::element::from<OutT>());
    });

    return static_cast<OutT>(value);
}

template <typename OutT, typename InT>
enable_t<OutT, std::is_integral<InT>, std::is_signed<InT>, std::is_integral<OutT>, std::is_unsigned<OutT>> checked_cast(
        InT value) {
    OPENVINO_ASSERT(value >= 0, "Can not safely cast ", static_cast<int64_t>(value), " from ", ov::element::from<InT>(),
                    " to ", ov::element::from<OutT>());

    details::staticIf<details::Greater(static_cast<std::make_unsigned_t<InT>>(std::numeric_limits<InT>::max()),
                                       std::numeric_limits<OutT>::max())>([&] {
        OPENVINO_ASSERT(static_cast<std::make_unsigned_t<InT>>(value) <= std::numeric_limits<OutT>::max(),
                        "Can not safely cast ", static_cast<int64_t>(value), " from ", ov::element::from<InT>(), " to ",
                        ov::element::from<OutT>());
    });

    return static_cast<OutT>(value);
}

template <typename OutT, typename InT>
enable_t<OutT, std::is_floating_point<InT>, std::is_integral<OutT>> checked_cast(InT value) {
    OPENVINO_ASSERT(value <= static_cast<InT>(std::numeric_limits<OutT>::max()), "Can not safely cast ", value,
                    " from ", ov::element::from<InT>(), " to ", ov::element::from<OutT>());

    OPENVINO_ASSERT(value >= static_cast<InT>(std::numeric_limits<OutT>::lowest()), "Can not safely cast ", value,
                    " from ", ov::element::from<InT>(), " to ", ov::element::from<OutT>());

    return static_cast<OutT>(value);
}

template <typename OutT, typename InT>
enable_t<OutT, std::is_integral<InT>, std::is_signed<InT>, std::is_floating_point<OutT>> checked_cast(InT value) {
    OPENVINO_ASSERT(static_cast<InT>(static_cast<OutT>(value)) == value, "Can not safely cast ",
                    static_cast<int64_t>(value), " from ", ov::element::from<InT>(), " to ", ov::element::from<OutT>());

    return static_cast<OutT>(value);
}

template <typename OutT, typename InT>
enable_t<OutT, std::is_integral<InT>, std::is_unsigned<InT>, std::is_floating_point<OutT>> checked_cast(InT value) {
    OPENVINO_ASSERT(static_cast<InT>(static_cast<OutT>(value)) == value, "Can not safely cast ",
                    static_cast<uint64_t>(value), " from ", ov::element::from<InT>(), " to ",
                    ov::element::from<OutT>());

    return static_cast<OutT>(value);
}

template <typename OutT, typename InT>
enable_t<OutT, std::is_same<double, InT>, std::is_same<float, OutT>> checked_cast(InT value) {
    OPENVINO_ASSERT(static_cast<InT>(static_cast<OutT>(value)) == value, "Can not safely cast ", value, " from ",
                    ov::element::from<InT>(), " to ", ov::element::from<OutT>());

    return static_cast<OutT>(value);
}

template <typename OutT, typename InT>
enable_t<OutT, std::is_same<float, InT>, std::is_same<double, OutT>> checked_cast(InT value) {
    return static_cast<OutT>(value);
}

//
// Custom float types
//

template <typename OutT>
enable_t<OutT, std::is_same<ov::float16, OutT>> checked_cast(ov::bfloat16 val) {
    return ov::float16(static_cast<float>(val));
}
template <typename OutT>
enable_t<OutT, std::is_same<ov::float8_e4m3, OutT>> checked_cast(ov::bfloat16 val) {
    return ov::float8_e4m3(static_cast<float>(val));
}
template <typename OutT>
enable_t<OutT, std::is_same<ov::float8_e5m2, OutT>> checked_cast(ov::bfloat16 val) {
    return ov::float8_e5m2(static_cast<float>(val));
}
template <typename OutT>
enable_t<OutT, std::is_same<ov::bfloat16, OutT>> checked_cast(ov::float16 val) {
    return ov::bfloat16(static_cast<float>(val));
}
template <typename OutT>
enable_t<OutT, std::is_same<ov::float8_e4m3, OutT>> checked_cast(ov::float16 val) {
    return ov::float8_e4m3(static_cast<float>(val));
}
template <typename OutT>
enable_t<OutT, std::is_same<ov::float8_e5m2, OutT>> checked_cast(ov::float16 val) {
    return ov::float8_e5m2(static_cast<float>(val));
}
template <typename OutT>
enable_t<OutT, std::is_same<ov::float8_e5m2, OutT>> checked_cast(ov::float8_e4m3 val) {
    return ov::float8_e5m2(static_cast<float>(val));
}
template <typename OutT>
enable_t<OutT, std::is_same<ov::float16, OutT>> checked_cast(ov::float8_e4m3 val) {
    return ov::float16(static_cast<float>(val));
}
template <typename OutT>
enable_t<OutT, std::is_same<ov::bfloat16, OutT>> checked_cast(ov::float8_e4m3 val) {
    return ov::bfloat16(static_cast<float>(val));
}
template <typename OutT>
enable_t<OutT, std::is_same<ov::float8_e4m3, OutT>> checked_cast(ov::float8_e5m2 val) {
    return ov::float8_e4m3(static_cast<float>(val));
}
template <typename OutT>
enable_t<OutT, std::is_same<ov::float16, OutT>> checked_cast(ov::float8_e5m2 val) {
    return ov::float16(static_cast<float>(val));
}
template <typename OutT>
enable_t<OutT, std::is_same<ov::bfloat16, OutT>> checked_cast(ov::float8_e5m2 val) {
    return ov::bfloat16(static_cast<float>(val));
}

template <typename OutT>
enable_t<OutT, not_<or_<std::is_same<ov::float8_e4m3, OutT>, std::is_same<ov::float8_e5m2, OutT>,
                        std::is_same<ov::float16, OutT>>>>
checked_cast(ov::bfloat16 val) {
    return checked_cast<OutT>(static_cast<float>(val));
}
template <typename OutT>
enable_t<OutT, not_<or_<std::is_same<ov::float8_e4m3, OutT>, std::is_same<ov::float8_e5m2, OutT>,
                        std::is_same<ov::bfloat16, OutT>>>>
checked_cast(ov::float16 val) {
    return checked_cast<OutT>(static_cast<float>(val));
}
template <typename OutT>
enable_t<OutT, not_<or_<std::is_same<ov::float8_e5m2, OutT>, std::is_same<ov::bfloat16, OutT>,
                        std::is_same<ov::float16, OutT>>>>
checked_cast(ov::float8_e4m3 val) {
    return checked_cast<OutT>(static_cast<float>(val));
}
template <typename OutT>
enable_t<OutT, not_<or_<std::is_same<ov::float8_e4m3, OutT>, std::is_same<ov::bfloat16, OutT>,
                        std::is_same<ov::float16, OutT>>>>
checked_cast(ov::float8_e5m2 val) {
    return checked_cast<OutT>(static_cast<float>(val));
}

template <typename OutT, typename InT>
enable_t<OutT, std::is_same<ov::bfloat16, OutT>> checked_cast(InT val) {
    return ov::bfloat16(checked_cast<float>(val));
}
template <typename OutT, typename InT>
enable_t<OutT, std::is_same<ov::float16, OutT>> checked_cast(InT val) {
    return ov::float16(checked_cast<float>(val));
}
template <typename OutT, typename InT>
enable_t<OutT, std::is_same<ov::float8_e4m3, OutT>> checked_cast(InT val) {
    return ov::float8_e4m3(checked_cast<float>(val));
}
template <typename OutT, typename InT>
enable_t<OutT, std::is_same<ov::float8_e5m2, OutT>> checked_cast(InT val) {
    return ov::float8_e5m2(checked_cast<float>(val));
}

//
// Wrapper
//

template <typename OutT, typename InT>
OutT convertValuePrecision(InT value) {
    return checked_cast<OutT>(value);
}

}  // namespace utils
}  // namespace npu
