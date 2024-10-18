// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifdef OV_CORE_USE_INTRINSICS
#    include <immintrin.h>

#    include "openvino/reference/utils/convert_util.hpp"

namespace ov {
namespace reference {
#    ifdef HAVE_AVX2

// Clamp optimized specializations
template <>
__m128i NoClamp::apply<__m256i, __m128i>(const __m256i vec_i32);

template <>
template <>
__m256 Clamp<float, float16>::apply<__m256, __m256>(const __m256 vec_f32);

// Conversion optimized specializations
// --- f32 -> other
template <>
template <>
struct Converter<float, float16>::Optimized<NoClamp> {
    static constexpr bool enabled = true;
    static void run(const float* in, float16* out);
};

template <>
template <>
struct Converter<float, float16>::Optimized<Clamp<float, float16>> {
    static constexpr bool enabled = true;
    static void run(const float* in, float16* out);
};

template <>
template <>
struct Converter<float, int8_t>::Optimized<NoClamp> {
    static constexpr bool enabled = true;
    static void run(const float* in, int8_t* out);
};

// --- f16 -> other
template <>
template <>
struct Converter<float16, float>::Optimized<NoClamp> {
    static constexpr bool enabled = true;
    static void run(const float16* in, float* out);
};

template <>
template <>
struct Converter<float16, int8_t>::Optimized<NoClamp> {
    static constexpr bool enabled = true;
    static void run(const float16* in, int8_t* out);
};

// --- bf16 -> other
template <>
template <>
struct Converter<bfloat16, float16>::Optimized<Clamp<float, float16>> {
    static constexpr bool enabled = true;
    static void run(const bfloat16* in, float16* out);
};

template <>
template <>
struct Converter<bfloat16, float>::Optimized<NoClamp> {
    static constexpr bool enabled = true;
    static void run(const bfloat16* in, float* out);
};

// --- u8 -> other
template <>
template <>
struct Converter<uint8_t, float16>::Optimized<NoClamp> {
    static constexpr bool enabled = true;
    static void run(const uint8_t* in, float16* out);
};
#    endif  // HAVE_AVX2
}  // namespace reference
}  // namespace ov
#endif
