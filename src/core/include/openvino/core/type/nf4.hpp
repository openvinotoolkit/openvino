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


namespace ov {
class OPENVINO_API ConvertNF4 {
public:
    constexpr ConvertNF4() = default;

    template <typename T, typename std::enable_if<!std::is_integral<T>::value, bool>::type = true>
    static void pack(uint8_t* dst, T* src, std::size_t idx) {
        uint8_t val = quantize_nf4(static_cast<float>(src[idx]));
        const size_t byte_idx = idx / 2;
        const uint8_t bit_shift = 4 * (idx % 2);
        dst[byte_idx] &= ~(0xF << bit_shift);         // half byte zeroed
        dst[byte_idx] |= ((val & 0xF) << bit_shift);  // set 1's
    }

    template <typename T, typename std::enable_if<std::is_integral<T>::value, bool>::type = true>
    static void pack(uint8_t* dst, T* src, std::size_t idx) {
        uint8_t val = static_cast<uint8_t>(src[idx]);
        const size_t byte_idx = idx / 2;
        const uint8_t bit_shift = 4 * (idx % 2);
        dst[byte_idx] &= ~(0xF << bit_shift);         // half byte zeroed
        dst[byte_idx] |= ((val & 0xF) << bit_shift);  // set 1's
    }

    template <typename T, typename std::enable_if<!std::is_integral<T>::value, bool>::type = true>
    static void unpack(T* dst, const uint8_t* src, std::size_t idx) {
        uint8_t nf4_idx = get_u4(src, idx);
        float val = dequantize_nf4(nf4_idx);
        dst[idx] = static_cast<T>(val);
    }

    template <typename T, typename std::enable_if<std::is_integral<T>::value, bool>::type = true>
    static void unpack(T* dst, const uint8_t* src, std::size_t idx) {
        uint8_t nf4_idx = get_u4(src, idx);
        dst[idx] = static_cast<T>(nf4_idx);
    }

    static float dequantize_nf4(uint8_t val);

    static uint8_t quantize_nf4(float x);

    static inline uint8_t get_u4(const uint8_t* buf, size_t idx) {
        const size_t byte_idx = idx / 2;
        const uint8_t bit_shift = 4 * (idx % 2);
        return (buf[byte_idx] >> bit_shift) & 0xF;
    }
};

};  // namespace ov
