// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "openvino/op/gelu.hpp"

namespace ov {
namespace reference {
template <typename T>
void gelu(const T* arg, T* out, op::GeluApproximationMode mode, size_t count) {
    if (mode == op::GeluApproximationMode::ERF) {
        for (size_t i = 0; i < count; i++) {
            out[i] = static_cast<T>((0.5 * arg[i] * (1 + erf(arg[i] / std::sqrt(2.0)))));
        }
    } else if (mode == op::GeluApproximationMode::TANH) {
        const auto pi = atan(1.0) * 4.0;
        const auto sqpi = std::sqrt(2.0 / pi);
        for (size_t i = 0; i < count; i++) {
            auto& x = arg[i];
            out[i] = static_cast<T>(0.5 * x * (1.0 + std::tanh(sqpi * (x + 0.044715 * std::pow(x, 3)))));
        }
    }
}
}  // namespace reference
}  // namespace ov
