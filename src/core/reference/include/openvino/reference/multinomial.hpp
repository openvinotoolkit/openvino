// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>

namespace ov {
namespace reference {
template <typename T>
void multinomial(const T* input, T* output, ov::element::Type output_type, bool replacement, bool log_probs, int64_t global_seed = 0, int64_t op_seed = 0) {
    for (size_t i = 0; i < count; i++) {
        out[i] = arg[i] < T(0) ? T(alpha * (std::exp(arg[i]) - 1.0)) : arg[i];
    }
}
}  // namespace reference
}  // namespace ov
