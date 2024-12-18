// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <type_traits>

namespace ov::intel_cpu {

#ifdef OPENVINO_CPP_23_VER
using to_underlying = std::to_underlying;
#else
// implementation of C++23 std::to_underlying
template <typename E>
constexpr auto to_underlying(E e) noexcept {
    return static_cast<std::underlying_type_t<E>>(e);
}
}  // namespace ov::intel_cpu
#endif
