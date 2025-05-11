// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>

namespace ov {
namespace reference {
template <typename T>
void selu(const T* arg,
          const T* alpha,
          const T* lambda,
          T* out,
          size_t size_arg,
          size_t size_alpha,
          size_t size_lambda) {
    for (size_t i = 0; i < size_arg; ++i) {
        out[i] = arg[i] > T(0) ? T(lambda[i % size_lambda] * arg[i])
                               : T(alpha[i % size_alpha] * lambda[i % size_lambda] * (std::exp(arg[i]) - 1));
    }
}
}  // namespace reference
}  // namespace ov
