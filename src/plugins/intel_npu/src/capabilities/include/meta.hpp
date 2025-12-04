//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <cstdint>
#include <type_traits>

namespace compat {
namespace meta {

// helper class to serve as C++17 std::void_t replacement
template <class...>
struct void_t {  // NOLINT: void_t name is intentionally the same as in C++17
    using type = void;
};

// IsMemCopyable<T>::value == true, if T "most likely" can be safely
// memcpy'ed to a byte array by code compiled with one compiler and
// then reinterpret_cast'ed back by another compiler.
// "Most likely" means there're still exceptions, under which T satisfies
// the criteria, but still can't be safely memcpy'ed, specifically:
// 1) T contains non-static data-members that are either pointers or references
// 2) T uses implicit padding
// while 2) can be mitigated via pragma pack(1) to ensure no implicit padding
// the only way to enforce 1) is code-review and custom static analysis tools
template <class, class = void>
struct IsMemCopyable : std::false_type {};

template <class T>
struct IsMemCopyable<T, typename void_t<
    typename std::enable_if<
        std::is_trivially_copyable<T>::value && std::is_standard_layout<T>::value
    >::type
>::type> : std::true_type {};

constexpr bool OPTIONAL = false;
constexpr bool REQUIRED = !OPTIONAL;

template <uint16_t id, bool policy = OPTIONAL>
struct CapabilityNo {
    static constexpr auto ID = id;
    static constexpr auto POLICY = policy;
};

template <class, class = void>
struct HasID : std::false_type {};

template <class T>
struct HasID<T,
    typename std::enable_if<
        std::is_same<decltype(T::ID), const uint16_t>::value
    >::type
> : std::true_type {};

template <class, class = void>
struct HasPolicy : std::false_type {};

template <class T>
struct HasPolicy<T,
    typename std::enable_if<
        std::is_same<decltype(T::POLICY), const bool>::value
    >::type
> : std::true_type {};

template <class, class = void>
struct HasMemberFunctionCheck : std::false_type {};

template <class T>
struct HasMemberFunctionCheck<T, typename void_t<
    typename std::enable_if<
        std::is_member_function_pointer<decltype(&T::isCompatible)>::value
    >::type,
    decltype(static_cast<bool(T::*)(const T&) const>(&T::isCompatible))
>::type> : std::true_type {};

template <class, class = void>
struct HasStaticMemberFunctionCheck : std::false_type {};

template <class T>
struct HasStaticMemberFunctionCheck<T, typename void_t<
    typename std::enable_if<
        !std::is_member_function_pointer<decltype(&T::isCompatible)>::value
    >::type,
    decltype(static_cast<bool(*)(const T&)>(&T::isCompatible))
>::type> : std::true_type {};

// IsCapability<T>::value == true, if and only if
// 1) IsMemCopyable<T>::value == true
// 2) T has static constexpr/const data-member ID of type uint16_t
// 3) T has static constexpr/const data-member POLICY of type bool
// 4) T is either
//  4.1) empty and has no member-function isCompatible, or
//  4.2) non-empty and has non-static const member-function with name isCompatible
//       that takes l-value reference to const T and returns bool, or
//  4.3) non-empty and has static member-function with name isCompatible
//       that takes l-value reference to const T and returns bool

template <class, class = void>
struct IsCapability : std::false_type {};

template <class T>
struct IsCapability<T, typename void_t<
    typename std::enable_if<
        HasID<T>::value &&
        HasPolicy<T>::value &&
        ((std::is_empty<T>::value && !HasMemberFunctionCheck<T>::value && !HasStaticMemberFunctionCheck<T>::value) ||
        (!std::is_empty<T>::value && (HasMemberFunctionCheck<T>::value || HasStaticMemberFunctionCheck<T>::value)))
    >::type
>::type> : std::true_type {};

}  // namespace meta
}  // namespace compat
