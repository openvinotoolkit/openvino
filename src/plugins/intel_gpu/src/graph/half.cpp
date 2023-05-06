// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stdint.h>

#ifdef HAVE_SSE
#include <immintrin.h>
#else
#include "openvino/core/type/float16.hpp"
#endif // HAVE_SSE

#include "intel_gpu/runtime/half.hpp"

namespace cldnn {

#ifdef HAVE_SSE

float half_to_float(uint16_t value) {
    static const uint32_t FLOAT16_EXP_SHIFT = (23 - 10);
    static const uint32_t FLOAT16_EXP_MASK = 0x7C00;
    static const uint32_t FLOAT32_EXP_MASK = 0x7F800000;
    static const uint32_t FLOAT16_MANTISSA_MASK = 0x03FF;
    static const uint32_t FLOAT16_TO_32_BIAS_DIFF_DENORM =
        ((127 - 15 - 10)
         << 23);  // The difference is (127-15) but we want to do the calculation in the exp place (bit 23:32)
    static const uint32_t FLOAT16_TO_32_BIAS_DIFF = ((127 - 15) << 10);
    static const uint32_t FLOAT16_IMPLICIT_1 = (1 << 10);
    static const uint32_t FLOAT16_EXP_MIN = (1 << 10);
    static const uint32_t FLOAT16_SIGN_MASK = 0x8000;
    __m128i a = _mm_unpacklo_epi16(_mm_set1_epi16(value), _mm_setzero_si128());
    __m128i exps = _mm_and_si128(_mm_set1_epi32(FLOAT16_EXP_MASK), a);           // Mask the exponents
    __m128i mantissa = _mm_and_si128(_mm_set1_epi32(FLOAT16_MANTISSA_MASK), a);  // Mask the mantissa
    __m128i signs = _mm_and_si128(_mm_set1_epi32(FLOAT16_SIGN_MASK), a);
    signs = _mm_slli_epi32(signs, 16);

    __m128i nans = _mm_cmpeq_epi32(exps, _mm_set1_epi32(FLOAT16_EXP_MASK));
    nans = _mm_and_si128(nans, _mm_set1_epi32(FLOAT32_EXP_MASK));
    nans = _mm_or_si128(nans, signs);

    __m128i subnormals = _mm_cmpeq_epi32(exps, _mm_setzero_si128());

    int out32;
    // e\m| 0 | 1
    // ------------
    //  0 | 0 | S
    // ------------
    //  1 | N | N
    //
    // The expression: (~exp) & mantissa, will evaluate to 0 exactly when the number is non subnormal or it's zero (just
    // like in the table) testz Tests for this condition
    if (_mm_testz_si128(subnormals, mantissa)) {
        __m128i tmp;
        exps = _mm_add_epi32(exps, _mm_set1_epi32(FLOAT16_TO_32_BIAS_DIFF));
        tmp = _mm_or_si128(exps, mantissa);
        tmp = _mm_slli_epi32(tmp, FLOAT16_EXP_SHIFT);
        tmp = _mm_blendv_epi8(
            tmp,
            _mm_setzero_si128(),
            subnormals);  // The idea is of course to use blendv_ps, but epi8 will work the same and won't switch stack
        tmp = _mm_or_si128(tmp, nans);
        out32 = _mm_extract_epi32(tmp, 0);
    } else {
        __m128i normals = _mm_andnot_si128(subnormals, _mm_set1_epi32(FLOAT16_IMPLICIT_1));  // Mark all normal numbers
        mantissa = _mm_or_si128(mantissa, normals);                                          // Apply implicit bit
        exps = _mm_max_epi16(
            exps,
            _mm_set1_epi32(
                FLOAT16_EXP_MIN));  // All subnormals will have 1 in the exponent (needed for correct bias computation)
        exps = _mm_slli_epi32(exps, FLOAT16_EXP_SHIFT);
        exps = _mm_add_epi32(exps, _mm_set1_epi32(FLOAT16_TO_32_BIAS_DIFF_DENORM));
        __m128 tmp;
        tmp = _mm_mul_ps(_mm_castsi128_ps(exps), _mm_cvtepi32_ps(mantissa));
        tmp = _mm_or_ps(tmp, _mm_castsi128_ps(nans));
        out32 = _mm_extract_ps(tmp, 0);
    }

    float outf32 = *reinterpret_cast<float*>(&out32);
    return outf32;
}

uint16_t float_to_half(float value) {
#define TO_M128i(a) (*reinterpret_cast<__m128i*>(&(a)))
#define TO_M128(a) (*const_cast<__m128*>(reinterpret_cast<const __m128*>(&(a))))

    static const uint32_t DWORD_SIGNMASK = 0x80000000;
    static const uint32_t DWORD_MINFP16 = 0x38800000;
    static const uint32_t DWORD_MAXFP16 = 0x477fe000;
    static const uint32_t DWORD_FP16_2_POW_10 = (1 << 10);
    static const uint32_t DWORD_FP16_EXPBIAS_NO_HALF = 0xc8000000;
    static const uint32_t WORD_MAXFP16 = 0x7BFF;

    static const __m128i IVec4SignMask = _mm_set1_epi32(DWORD_SIGNMASK);
    static const __m128i IVec4MinNormalFp16 = _mm_set1_epi32(DWORD_MINFP16);
    static const __m128i IVec4MaxNormalFp16 = _mm_set1_epi32(DWORD_MAXFP16);
    static const __m128i IVec4OnePow10 = _mm_set1_epi32(DWORD_FP16_2_POW_10);
    static const __m128i IVec4ExpBiasFp16 = _mm_set1_epi32(DWORD_FP16_EXPBIAS_NO_HALF);
    static const __m128i IVec4MaxFp16InWords = _mm_set1_epi32(WORD_MAXFP16);

    static const __m128 FVec4MaxNormalFp16 = TO_M128(IVec4MaxNormalFp16);
    static const __m128 FVec4MinNormalFp16 = TO_M128(IVec4MinNormalFp16);
    static const __m128i IVec4InfF32 = _mm_set1_epi32(0x7f800000);  // inf in in hex representation
    static const __m128i IVec4InfF16 = _mm_set1_epi32(0x00007c00);

    static const __m128 FVec4MaxFp16InWords = TO_M128(IVec4MaxFp16InWords);

    __m128 Src = _mm_set1_ps(value);

    // Remove the sign bit from the source
    __m128 AbsSrc = _mm_andnot_ps(TO_M128(IVec4SignMask), Src);

    // Create a mask to identify the DWORDs that are smaller than the minimum normalized fp16 number
    __m128 CmpToMinFp16Mask = _mm_cmplt_ps(AbsSrc, FVec4MinNormalFp16);

    // Create a mask to identify the DWORDs that are larger than the maximum normalized fp16 number
    __m128 CmpToMaxFp16Mask = _mm_cmpgt_ps(AbsSrc, FVec4MaxNormalFp16);
    __m128i CmpToInfMask = _mm_cmpeq_epi32(TO_M128i(AbsSrc), IVec4InfF32);
    // Create a mask with the minimum normalized fp16 number in the DWORDs that are smaller than it
    __m128 MaskOfMinFp16 = _mm_and_ps(CmpToMinFp16Mask, FVec4MinNormalFp16);

    __m128i MaskOf2POW10 = _mm_and_si128(TO_M128i(CmpToMinFp16Mask), IVec4OnePow10);
    __m128 ResultPS = _mm_add_ps(AbsSrc, MaskOfMinFp16);
    __m128i Result = TO_M128i(ResultPS);

    // We need to move from a 127 biased domain to a 15 biased domain. This means subtracting 112 from the exponent. We
    // will add '-112' to the exponent but since the exponent is shifted 23 bits to the left we need to shift '-112' 23
    // bits to the left as well. This gives us 0xC8000000. We are going to shift the mantissa 13 bits to the right
    // (moving from 23 bits mantissa to 10).
    Result = _mm_add_epi32(Result, IVec4ExpBiasFp16);

    // Shift the mantissa to go from 23 bits to 10 bits
    Result = _mm_srli_epi32(Result, 13);

    Result = _mm_sub_epi16(Result, MaskOf2POW10);

    ResultPS = _mm_blendv_ps(TO_M128(Result), FVec4MaxFp16InWords, CmpToMaxFp16Mask);
    Result = TO_M128i(ResultPS);
    // infinity preserving blending
    Result = _mm_blendv_epi8(Result, IVec4InfF16, CmpToInfMask);

    __m128i iPackedResult = _mm_packs_epi32(Result, Result);

    // iSignMask = mask of the sign bits of the source 4 dwords
    __m128i iSignMask = _mm_and_si128(TO_M128i(Src), IVec4SignMask);

    // Pack the sign mask to 4 words
    __m128i iSignInWords = _mm_packs_epi32(iSignMask, iSignMask);

    iPackedResult = _mm_or_si128(iPackedResult, iSignInWords);
    return (uint16_t)_mm_extract_epi16(iPackedResult, 0);
}

#else

float half_to_float(uint16_t value) {
    return ov::float16(value);
}

uint16_t float_to_half(float value) {
    return ov::float16(value);
}

#endif // HAVE_SSE

}  // namespace cldnn
