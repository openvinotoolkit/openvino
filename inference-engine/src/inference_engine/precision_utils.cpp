// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "precision_utils.h"

#include <stdint.h>

namespace InferenceEngine {
namespace PrecisionUtils {

void f16tof32Arrays(float* dst, const short* src, size_t nelem, float scale, float bias) {
    const ie_fp16* _src = reinterpret_cast<const ie_fp16*>(src);

    for (size_t i = 0; i < nelem; i++) {
        dst[i] = PrecisionUtils::f16tof32(_src[i]) * scale + bias;
    }
}

void f32tof16Arrays(short* dst, const float* src, size_t nelem, float scale, float bias) {
    for (size_t i = 0; i < nelem; i++) {
        dst[i] = PrecisionUtils::f32tof16(src[i] * scale + bias);
    }
}

// Function to convert F32 into F16
// F32: exp_bias:127 SEEEEEEE EMMMMMMM MMMMMMMM MMMMMMMM.
// F16: exp_bias:15  SEEEEEMM MMMMMMMM
#define EXP_MASK_F32 0x7F800000U
#define EXP_MASK_F16 0x7C00U

// small helper function to represent uint32_t value as float32
inline float asfloat(uint32_t v) {
    // Both type-punning casts and unions are UB per C++ spec
    // But compilers usually only break code with casts
    union {
        float f;
        uint32_t i;
    };
    i = v;
    return f;
}

// Function to convert F16 into F32
float f16tof32(ie_fp16 x) {
    // this is storage for output result
    uint32_t u = static_cast<uint32_t>(x);

    // get sign in 32bit format
    uint32_t s = ((u & 0x8000) << 16);

    // check for NAN and INF
    if ((u & EXP_MASK_F16) == EXP_MASK_F16) {
        // keep mantissa only
        u &= 0x03FF;

        // check if it is NAN and raise 10 bit to be align with intrin
        if (u) {
            u |= 0x0200;
        }

        u <<= (23 - 10);
        u |= EXP_MASK_F32;
        u |= s;
    } else if ((u & EXP_MASK_F16) == 0) {  // check for zero and denormals.
        uint16_t h_sig = (u & 0x03ffu);
        if (h_sig == 0) {
            /* Signed zero */
            u = s;
        } else {
            /* Subnormal */
            uint16_t h_exp = (u & EXP_MASK_F16);
            h_sig <<= 1;
            while ((h_sig & 0x0400u) == 0) {
                h_sig <<= 1;
                h_exp++;
            }
            uint32_t f_exp = (static_cast<uint32_t>(127 - 15 - h_exp)) << 23;
            uint32_t f_sig = (static_cast<uint32_t>(h_sig & 0x03ffu)) << 13;
            u = s + f_exp + f_sig;
        }
    } else {
        // abs
        u = (u & 0x7FFF);

        // shift mantissa and exp from f16 to f32 position
        u <<= (23 - 10);

        // new bias for exp (f16 bias is 15 and f32 bias is 127)
        u += ((127 - 15) << 23);

        // add sign
        u |= s;
    }

    // finaly represent result as float and return
    return asfloat(u);
}

// This function convert f32 to f16 with rounding to nearest value to minimize error
// the denormal values are converted to 0.
ie_fp16 f32tof16(float x) {
    // create minimal positive normal f16 value in f32 format
    // exp:-14,mantissa:0 -> 2^-14 * 1.0
    static float min16 = asfloat((127 - 14) << 23);

    // create maximal positive normal f16 value in f32 and f16 formats
    // exp:15,mantissa:11111 -> 2^15 * 1.(11111)
    static float max16 = asfloat(((127 + 15) << 23) | 0x007FE000);
    static uint32_t max16f16 = ((15 + 15) << 10) | 0x3FF;

    // define and declare variable for intermediate and output result
    // the union is used to simplify representation changing
    union {
        float f;
        uint32_t u;
    } v;
    v.f = x;

    // get sign in 16bit format
    uint32_t s = (v.u >> 16) & 0x8000;  // sign 16:  00000000 00000000 10000000 00000000

    // make it abs
    v.u &= 0x7FFFFFFF;  // abs mask: 01111111 11111111 11111111 11111111

    // check NAN and INF
    if ((v.u & EXP_MASK_F32) == EXP_MASK_F32) {
        if (v.u & 0x007FFFFF) {
            return s | (v.u >> (23 - 10)) | 0x0200;  // return NAN f16
        } else {
            return s | EXP_MASK_F16;  // return INF f16
        }
    }

    // to make f32 round to nearest f16
    // create halfULP for f16 and add it to origin value
    float halfULP = asfloat(v.u & EXP_MASK_F32) * asfloat((127 - 11) << 23);
    v.f += halfULP;

    // if input value is not fit normalized f16 then return 0
    // denormals are not covered by this code and just converted to 0
    if (v.f < min16 * 0.5F) {
        return s;
    }

    // if input value between min16/2 and min16 then return min16
    if (v.f < min16) {
        return s | (1 << 10);
    }

    // if input value more than maximal allowed value for f16
    // then return this maximal value
    if (v.f >= max16) {
        return max16f16 | s;
    }

    // change exp bias from 127 to 15
    v.u -= ((127 - 15) << 23);

    // round to f16
    v.u >>= (23 - 10);

    return v.u | s;
}

}  // namespace PrecisionUtils
}  // namespace InferenceEngine
