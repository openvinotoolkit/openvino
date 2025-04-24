// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

namespace ov::util::bit {

/**
 * @brief Make empty bit mask, non of bit is set.
 *
 * @return 64-bit mask with none of bits set (0).
 */
template <class = void>
constexpr uint64_t mask() {
    return 0;
}

/**
 * @brief Makes bit mask with bits set at input positions.
 *
 * @tparam T          Type of bit position.
 * @tparam Args       Type of other bit positions.
 *
 * @param bit_pos     Bit position to set.
 * @param other_bits  Other bit positions to set.
 * @return 64-bit mask.
 */
template <class T, class... Args>
constexpr uint64_t mask(T bit_pos, Args... other_bits) {
    return mask(other_bits...) | (static_cast<uint64_t>(1) << bit_pos);
}

}  // namespace ov::util::bit
