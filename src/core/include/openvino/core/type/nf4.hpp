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

    template<typename T>
    static void pack(uint8_t *dst, const T *src, std::size_t count);

    template<typename T>
    static void pack_one(uint8_t *dst, T *src, std::size_t idx);

    template<typename T>
    static void unpack(T *dst, uint8_t *src, std::size_t count);

    template<typename TO, typename TI>
    static void unpack_one(TO *dst, TI src, std::size_t idx) {
        uint8_t nf4_idx = static_cast<uint8_t>(src);
        dst[idx] = static_cast<TO>(ConvertNF4::dDequantizeNF4(nf4_idx));
    }
private:
    static float dDequantizeNF4(uint8_t val);

    static uint8_t dQuantizeNF4(float x);
};


template<typename T>
void ConvertNF4::pack_one(uint8_t *dst, T *src, std::size_t idx)
{
    uint8_t val = dQuantizeNF4(static_cast<float>(src[idx]));
    const size_t byte_idx = idx / 2;
    const uint8_t bit_shift = 4 * (++idx % 2);
    dst[byte_idx] &= ~(0xF << bit_shift);         // half byte zeroed
    dst[byte_idx] |= ((val & 0xF) << bit_shift);  // set 1's
}


template <typename T>
void ConvertNF4::pack(uint8_t *dst, const T *src, std::size_t count) {
    for (size_t i = 0; i < count; i++) {
        pack_one(dst, src, i);
    }
}

template <typename T>
void ConvertNF4::unpack(T *dst, uint8_t *src, std::size_t count) {
    for (size_t i = 0; i < count; i++) {
        unpack_one(dst, src, i);
    }
}

};
