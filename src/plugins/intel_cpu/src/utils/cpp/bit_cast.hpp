// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstring>
#include <type_traits>
#if defined(OPENVINO_CPP_VER_AT_LEAST_20)
#    include <bit>
#endif

namespace ov::intel_cpu {

#if defined(OPENVINO_CPP_VER_AT_LEAST_20)
using std::bit_cast;
#else
template <typename To, typename From>
inline std::enable_if_t<sizeof(To) == sizeof(From) && std::is_trivially_copyable_v<From> &&
                            std::is_trivially_copyable_v<To>,
                        To>
bit_cast(const From& src) noexcept {
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
