// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cfenv>
#include <cmath>
#include <type_traits>

#include "openvino/op/round.hpp"
#include "openvino/reference/rounding_guard.hpp"
#include "openvino/reference/utils/type_util.hpp"

namespace ov {
namespace reference {
/**
 * @brief Rounding algorithm for ov::op::v5::Round::RoundMode::HALF_TO_EVEN.
 *
 * @tparam T     Value type.
 * @param value  Value for rounding.
 * @return       Rounded value.
 */
template <typename T>
T round_to_nearest_even(T value) {
    return std::nearbyint(value);
}

/**
 * @brief Rounding algorithm for ov::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO.
 *
 * @tparam T     Value type.
 * @param value  Value for rounding.
 * @return       Rounded value.
 */
template <typename T>
T round_half_away_zero(T value) {
    return std::round(value);
}

/**
 * @brief Reference implementation of Round operator.
 *
 * Used when T is OpenVINO floating type.
 *
 * @param arg    Input buffer pointer with data to round.
 * @param out    Output buffer pointer with rounded results.
 * @param count  Number of elements in input tensor.
 * @param mode   Rounding mode.
 */
template <typename T, typename std::enable_if<ov::is_floating_point<T>()>::type* = nullptr>
void round(const T* arg, T* out, const size_t count, const op::v5::Round::RoundMode mode) {
    const ov::RoundingGuard round_g{FE_TONEAREST};
    const auto round_algo =
        (mode == op::v5::Round::RoundMode::HALF_TO_EVEN) ? round_to_nearest_even<T> : round_half_away_zero<T>;

    std::transform(arg, arg + count, out, round_algo);
}

/**
 * @brief Reference implementation of Round operator.
 *
 * Used when T is OpenVINO integral type.
 *
 * @param arg    Input buffer pointer with data to round.
 * @param out    Output buffer pointer with rounded results.
 * @param count  Number of elements in input tensor.
 * @param mode   Rounding mode.
 */
template <typename T, typename std::enable_if<!ov::is_floating_point<T>()>::type* = nullptr>
void round(const T* arg, T* out, const size_t count, const op::v5::Round::RoundMode) {
    std::copy_n(arg, count, out);
}
}  // namespace reference
}  // namespace ov
