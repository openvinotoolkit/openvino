// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <functional>
#include <sstream>
#include <iomanip>
#include <limits>
#include <algorithm>
#include "llm_types.hpp"
#include "tensor2d.hpp"
#include "bf16.hpp"
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <stdlib.h>

// https://stackoverflow.com/questions/570669/checking-if-a-double-or-float-is-nan-in-c/57770634#57770634
static inline uint32_t load_ieee754_rep(float a) {
    uint32_t r;
    static_assert(sizeof r == sizeof a, "Unexpected sizes.");
    std::memcpy(&r, &a, sizeof a); // Generates movd instruction.
    return r;
}
constexpr uint32_t inf_float_shl1 = UINT32_C(0xff000000);
// The shift left removes the sign bit. The exponent moves into the topmost bits,
// so that plain unsigned comparison is enough.
static inline bool isnan2(float a)     { return load_ieee754_rep(a) << 1  > inf_float_shl1; }
static inline bool isinf2(float a)     { return load_ieee754_rep(a) << 1 == inf_float_shl1; }
static inline bool isfinite2(float a)  { return load_ieee754_rep(a) << 1  < inf_float_shl1; }

template<typename T>
void fill_rnd(tensor2D<T>& t) {
    auto * p = t.data;
    int i = 0;
    int total = t.dims[0] * t.padded_dim1;
    // +1 -1 for integer types
    // 0.5 -0.5 for float point 
    float scale = std::is_integral<T>::value ? 2:1;
    for(i = 0; i + 8 <= total; i+=8) {
        // lower mantissa can help to avoid small errors in accuracy comparison
        auto num = rand() & 0xFF;
        p[i]   = scale*((num & 1) - 0.5f); num>>=1;
        p[i+1] = scale*((num & 1) - 0.5f); num>>=1;
        p[i+2] = scale*((num & 1) - 0.5f); num>>=1;
        p[i+3] = scale*((num & 1) - 0.5f); num>>=1;
        p[i+4] = scale*((num & 1) - 0.5f); num>>=1;
        p[i+5] = scale*((num & 1) - 0.5f); num>>=1;
        p[i+6] = scale*((num & 1) - 0.5f); num>>=1;
        p[i+7] = scale*((num & 1) - 0.5f); num>>=1;
    }
    for(; i<total; i++) {
        auto num = rand();
        p[i] = scale*((num & 1) - 0.5f);
    }
}

template<typename T>
bool operator==(const tensor2D<T>& lhs, const tensor2D<T>& rhs) {
    if (lhs.dims[0] != rhs.dims[0] || lhs.dims[1] != rhs.dims[1])
        return false;
    for(int i0 = 0; i0 < lhs.dims[0]; i0++)
    for(int i1 = 0; i1 < lhs.dims[1]; i1++) {
        // with -ffast-math,  std::isnan, std::isinf,  x != x  always return false
        // so we need special logic to test nan here
        if (std::is_same<T, ov::bfloat16>::value ||
            std::is_same<T, float>::value) {
            float f0 = lhs(i0,i1);
            float f1 = rhs(i0,i1);
            if (isnan2(f1) || isnan2(f0)) {
                std::cout << " nan is found: f0=" << f0 << ",  f1=" << f1 << std::endl;
                return false;
            }
            if (std::abs(f0 - f1) <= 0.01)
                continue;
        }

        if (lhs(i0,i1) == rhs(i0,i1))
            continue;
        std::cout << " operator== failed at (" << i0 << ", " << i1 << ")  value "
                    << lhs(i0,i1) << "!=" << rhs(i0,i1) << std::endl;
        return false;
    }
    return true;
}

template<typename T>
bool is_normal(const tensor2D<T>& t) {
    for (int i0 = 0; i0 < t.dims[0]; i0++)
    for (int i1 = 0; i1 < t.dims[1]; i1++) {
        float f0 = t(i0,i1);
        if (isnan2(f0)) {
            std::cout << " found nan at (" << i0 << "," << i1 << ")" << std::endl;
            return false;
        }
        if (isinf2(f0)) {
            std::cout << " found inf at (" << i0 << "," << i1 << ")" << std::endl;
            return false;
        }
    }
    return true;
}

template<typename T>
bool compare(const tensor2D<T>& lhs, const tensor2D<T>& rhs, float tolerance) {
    float max_abs_diff = 0;
    float max_rel_diff = 0;
    if (lhs.dims[0] != rhs.dims[0] || lhs.dims[1] != rhs.dims[1])
        return false;
    for (int i0 = 0; i0 < lhs.dims[0]; i0++)
    for (int i1 = 0; i1 < lhs.dims[1]; i1++) {
        float f0 = lhs(i0, i1);
        float f1 = rhs(i0, i1);
        auto diff = std::fabs(f0 - f1);
        auto rel_diff = diff / std::fabs(f0);
        max_abs_diff = std::max(max_abs_diff, diff);
        if (std::fabs(lhs(i0,i1) > 0) && diff > 0)
            max_rel_diff = std::max(max_rel_diff, rel_diff);
    }
    std::cout << "max_abs_diff=" << max_abs_diff << " max_rel_diff=" << max_rel_diff << "\n";
    return tolerance > max_abs_diff;
}

template<typename T>
std::ostream& operator<<(std::ostream& out, const tensor2D<T>& obj) {
    int i0;
    auto showline = [&](int i) {
        out << "[" << i << "," << 0 << "]: ";
        int i1;
        for(i1=0; i1<obj.dims[1] && i1 < 8; i1++) {
            out << +obj(i0,i1) << ",";
        }
        if (i1 < obj.dims[1]) out << "...";
        out << std::endl;
    };
    for(i0=0; i0 < obj.dims[0] && i0 < 32; i0++) {
        showline(i0);
    }
    if (i0 < obj.dims[0]) {
        out << "... ... ... ..." << std::endl;
        showline(obj.dims[0] - 1);
    }
    return out;
}

template<typename T>
inline void show(const T * data, int rows, int cols) {
    std::ostream& out = std::cout;
    out << "==============\n";
    for(int i0=0; i0 < rows; i0++) {
        out << "[" << i0 << "," << 0 << "]: ";
        for(int i1=0; i1<cols; i1++)
            //https://stackoverflow.com/questions/14644716/how-to-output-a-character-as-an-integer-through-cout/28414758#28414758
            out << +data[i0 * cols + i1] << ",";
        out << std::endl;
    }
}

template<typename T>
inline void vshow(__m512i v) {
    T values[512/8/sizeof(T)];
    _mm512_storeu_si512(values, v);
    show(values, 1, 512/8/sizeof(T));
}

template<typename T>
inline void vshow(__m512 v) {
    T values[512/8/sizeof(T)];
    _mm512_storeu_ps(values, v);
    show(values, 1, 512/8/sizeof(T));
}

template<typename T, int tile>
inline void tshow() {
    if (std::is_same<ov::bfloat16,T>::value) {
        ov::bfloat16 data[16*32];
        _tile_stored(tile, data, 64);
        show(data, 16, 32);
    }
    if (std::is_same<float,T>::value) {
        float data[16*16];
        _tile_stored(tile, data, 64);
        show(data, 16, 16);
    }
    if (std::is_same<int8_t,T>::value) {
        int8_t data[16*64];
        _tile_stored(tile, data, 64);
        show(data, 16, 64);
    }
    if (std::is_same<uint8_t,T>::value) {
        uint8_t data[16*64];
        _tile_stored(tile, data, 64);
        show(data, 16, 64);
    }
}
