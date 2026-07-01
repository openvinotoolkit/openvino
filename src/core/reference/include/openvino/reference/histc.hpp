// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "openvino/core/except.hpp"

namespace ov {
namespace reference {

template <typename T>
void histc(const T* input, size_t n, int64_t bins, double min_val, double max_val, T* output) {
    OPENVINO_ASSERT(bins >= 0, "Histc 'bins' attribute must be non-negative.");
    const auto bins_count = static_cast<size_t>(bins);

    std::fill(output, output + bins_count, T{0});

    if (n == 0 || bins == 0)
        return;

    double range_min = min_val;
    double range_max = max_val;
    if (range_min == 0.0 && range_max == 0.0) {
        range_min = static_cast<double>(*std::min_element(input, input + n));
        range_max = static_cast<double>(*std::max_element(input, input + n));
    }

    if (range_min == range_max) {
        for (size_t i = 0; i < n; ++i) {
            const double v = static_cast<double>(input[i]);
            if (v >= range_min && v <= range_max) {
                output[0] += T{1};
            }
        }
        return;
    }

    const double step = (range_max - range_min) / static_cast<double>(bins);
    for (size_t i = 0; i < n; ++i) {
        const double v = static_cast<double>(input[i]);
        if (v >= range_min && v <= range_max) {
            int64_t bin = static_cast<int64_t>((v - range_min) / step);
            if (bin >= bins)
                bin = bins - 1;
            output[bin] += T{1};
        }
    }
}

}  // namespace reference
}  // namespace ov
