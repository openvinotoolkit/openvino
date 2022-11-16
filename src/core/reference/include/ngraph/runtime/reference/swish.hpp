// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>

namespace ngraph {
namespace runtime {
namespace reference {
template <typename T>
void swish(const T* arg, const T* beta, T* out, size_t count) {
    T beta_value = static_cast<T>(1.0);
    if (beta != nullptr) {
        beta_value = beta[0];
    }
    for (size_t i = 0; i < count; i++) {
        out[i] = static_cast<T>(arg[i] / (1.0 + std::exp(-arg[i] * beta_value)));
    }
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
