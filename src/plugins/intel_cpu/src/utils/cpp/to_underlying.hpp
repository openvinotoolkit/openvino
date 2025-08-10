// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <type_traits>

namespace ov::intel_cpu {

#ifdef OPENVINO_CPP_VER_AT_LEAST_23
using to_underlying = std::to_underlying;
#else
// implementation of C++23 std::to_underlying
template <typename E>
constexpr auto to_underlying(E e) noexcept {
    return static_cast<std::underlying_type_t<E>>(e);
}
}  // namespace ov::intel_cpu
#endif
