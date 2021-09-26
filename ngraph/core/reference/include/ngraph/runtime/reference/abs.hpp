// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

namespace ngraph {
namespace runtime {
namespace reference {
template <typename T, typename std::enable_if<!std::is_integral<T>::value, bool>::type = true>
void abs(const T* arg, T* out, size_t count) {
    for (size_t i = 0; i < count; i++) {
        // TODO: generic "abs" doesn't work here for some reason.
        out[i] = (arg[i] < T(0) ? T(-arg[i]) : arg[i]);
    }
}

template<typename T, typename std::enable_if<std::is_integral<T>::value, bool>::type = true> 
void abs(const T* arg, T* out, size_t count) {
    for (size_t i = 0; i < count; i++) {
        out[i] = std::roundl((arg[i] < T(0) ? T(-arg[i]) : arg[i]));
    }
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
