// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

#include "openvino/core/except.hpp"

namespace ov::reference {

/**
 * @brief Reference implementation of Histc operator (compatible with torch.histc).
 *
 * @param data   Pointer to input data.
 * @param out    Pointer to output data with `bins` elements.
 * @param count  Number of input elements.
 * @param bins   Number of equal-width bins.
 * @param min    Lower end of the range. If equal to max, the range is taken from the data.
 * @param max    Upper end of the range. If equal to min, the range is taken from the data.
 */
template <class T>
void histc(const T* data, T* out, const size_t count, const size_t bins, double min, double max) {
    // Like torch.histc, f32 is binned in f32; integers are widened to keep fractional range ends.
    using C = std::conditional_t<std::is_same_v<T, double> || std::is_integral_v<T>, double, float>;

    if (min == max && count > 0) {
        // Like torch.histc, a NaN element makes the range NaN, which is rejected below.
        min = std::numeric_limits<double>::infinity();
        max = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < count; ++i) {
            const auto x = static_cast<double>(data[i]);
            if (std::isnan(x)) {
                min = max = x;
                break;
            }
            if (x < min)
                min = x;
            if (x > max)
                max = x;
        }
    }
    if (min == max) {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            min =
                std::min(static_cast<double>(std::nextafter(static_cast<T>(min), -std::numeric_limits<T>::infinity())),
                         min - 1.0);
            max = std::max(static_cast<double>(std::nextafter(static_cast<T>(max), std::numeric_limits<T>::infinity())),
                           max + 1.0);
        } else {
            min -= 1.0;
            max += 1.0;
        }
    }

    const auto lo = static_cast<C>(min);
    const auto hi = static_cast<C>(max);
    OPENVINO_ASSERT(std::isfinite(lo) && std::isfinite(hi), "Histc: range [", min, ", ", max, "] is not finite.");
    OPENVINO_ASSERT(lo < hi, "Histc: max must be larger than min. Got range [", min, ", ", max, "].");

    std::vector<size_t> hist(bins, 0);
    for (size_t i = 0; i < count; ++i) {
        const auto x = static_cast<C>(data[i]);
        if (!(x >= lo && x <= hi))
            continue;
        const auto idx = static_cast<size_t>((x - lo) * static_cast<C>(bins) / (hi - lo));
        ++hist[std::min(idx, bins - 1)];
    }
    std::transform(hist.begin(), hist.end(), out, [](size_t n) {
        if constexpr (std::is_integral_v<T>) {
            // Like torch.histc, the output keeps the input type and counts that do not fit into T wrap around.
            return static_cast<T>(n);
        } else {
            return static_cast<T>(static_cast<C>(n));
        }
    });
}
}  // namespace ov::reference
