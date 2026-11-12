// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <cstring>

#include "openvino/core/type/bfloat16.hpp"

namespace ov::test::utils {

template <typename T>
T ssm_reference_cast(float value, bool native_bf16 = true) {
    return static_cast<T>(value);
}

template <>
inline ov::bfloat16 ssm_reference_cast<ov::bfloat16>(float value, bool native_bf16) {
    if (!native_bf16) {
        return static_cast<ov::bfloat16>(value);
    }
    // Match native BF16 round-to-nearest-even without relying on the scalar constructor's rounding mode.
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7fffffffU) > 0x7f800000U) {
        return ov::bfloat16::from_bits(static_cast<uint16_t>((bits >> 16) | 0x0040U));
    }
    bits += 0x7fffU + ((bits >> 16) & 1U);
    return ov::bfloat16::from_bits(static_cast<uint16_t>(bits >> 16));
}

}  // namespace ov::test::utils
