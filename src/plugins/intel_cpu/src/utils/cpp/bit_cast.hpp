// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstring>
#include <type_traits>
#if defined(__has_include)
#    if __has_include(<version>)
#        include <version>
#    endif
#endif
#if defined(__cpp_lib_bit_cast)
#    include <bit>
#endif

namespace ov::intel_cpu {

// Gated on the library feature-test macro, not OPENVINO_CPP_VER_AT_LEAST_20: with
// Clang using an older libstdc++ (e.g. from GCC 10), the language standard can be
// C++20 while std::bit_cast is still missing from the linked standard library.
#if defined(__cpp_lib_bit_cast)
using std::bit_cast;
#else
template <typename To,
          typename From,
          typename = std::enable_if_t<sizeof(To) == sizeof(From) && std::is_trivially_copyable_v<From> &&
                                      std::is_trivially_copyable_v<To>>>
To bit_cast(const From& src) noexcept {
    static_assert(std::is_trivially_constructible_v<To>, "Destination type must be trivially constructible");
    To dst{};
#    if defined(__GNUC__) && !defined(__clang__)
#        pragma GCC diagnostic push
#        pragma GCC diagnostic ignored "-Wclass-memaccess"
#    endif
    std::memcpy(&dst, &src, sizeof(To));
#    if defined(__GNUC__) && !defined(__clang__)
#        pragma GCC diagnostic pop
#    endif
    return dst;
}
#endif

}  // namespace ov::intel_cpu
