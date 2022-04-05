// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <type_traits>

namespace ngraph {
namespace runtime {
namespace reference {
template <typename T, typename std::enable_if<std::is_unsigned<T>::value, bool>::type = true>
void abs(const T* arg, T* out, size_t count) {
    std::copy(arg, arg + count, out);
}

template <typename T, typename std::enable_if<!std::is_unsigned<T>::value, bool>::type = true>
void abs(const T* arg, T* out, size_t count) {
    for (size_t i = 0; i < count; i++) {
        // TODO: generic "abs" doesn't work here for some reason.
        out[i] = (arg[i] < T(0) ? T(-arg[i]) : arg[i]);
    }
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
