// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/type/float16.hpp"

#define ROUND_MODE_TO_NEAREST_EVEN

namespace ov {
class OPENVINO_API ConvertNF4 {
public:
    ConvertNF4() = default;

    template<typename T,
    typename std::enable_if<!std::is_integral<T>::value, bool>::type = true>
    static void pack(uint8_t *dst, T *src, std::size_t idx)
    {
        uint8_t val = dQuantizeNF4(static_cast<float>(src[idx]));
        const size_t byte_idx = idx / 2;
        const uint8_t bit_shift = 4 * (++idx % 2);
        dst[byte_idx] &= ~(0xF << bit_shift);         // half byte zeroed
        dst[byte_idx] |= ((val & 0xF) << bit_shift);  // set 1's
    }

    template<typename T,
    typename std::enable_if<std::is_integral<T>::value, bool>::type = true>
    static void pack(uint8_t *dst, T *src, std::size_t idx)
    {
        uint8_t val = static_cast<uint8_t>(src[idx]);
        const size_t byte_idx = idx / 2;
        const uint8_t bit_shift = 4 * (++idx % 2);
        dst[byte_idx] &= ~(0xF << bit_shift);         // half byte zeroed
        dst[byte_idx] |= ((val & 0xF) << bit_shift);  // set 1's
    }

    static float dDequantizeNF4(uint8_t val);

    static uint8_t dQuantizeNF4(float x);
};

};
