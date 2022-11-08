// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace ngraph {
namespace runtime {
namespace reference {
template <typename T, typename std::enable_if<!std::is_integral<T>::value, bool>::type = true>
void atanh(const T* arg, T* out, size_t count) {
    for (size_t i = 0; i < count; i++) {
        out[i] = std::atanh(arg[i]);
    }
}

template <typename T, typename std::enable_if<std::is_integral<T>::value, bool>::type = true>
void atanh(const T* arg, T* out, size_t count) {
    for (size_t i = 0; i < count; i++) {
        /**
         * Intgral type don't support: NAN and INFINITY.
         * So we clip input value, and make sure return avaiable value.
         */
        if (std::is_same<T, uint8_t>::value || std::is_same<T, uint32_t>::value || std::is_same<T, uint64_t>::value) {
            if (arg[i] > 0) {
                out[i] = std::numeric_limits<T>::max();
            } else {
                out[i] = static_cast<T>(std::roundl(std::atanh(arg[i])));
            }
        } else {
            if (arg[i] <= -1) {
                out[i] = std::numeric_limits<T>::min();
            } else if (arg[i] >= 1) {
                out[i] = std::numeric_limits<T>::max();
            } else {
                out[i] = static_cast<T>(std::roundl(std::atanh(arg[i])));
            }
        }
    }
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
