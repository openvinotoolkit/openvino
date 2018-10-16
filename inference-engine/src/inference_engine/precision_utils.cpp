// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "precision_utils.h"
#include <stdint.h>
#include <details/ie_exception.hpp>
#include <ie_blob.h>
#include <emmintrin.h>
#include <nmmintrin.h>
#include "inference_engine.hpp"

using namespace InferenceEngine;

void PrecisionUtils::f16tof32Arrays(float *dst, const short *src, size_t nelem, float scale, float bias) {
    const ie_fp16 *_src = reinterpret_cast<const ie_fp16 *>(src);

    for (size_t i = 0; i < nelem; i++) {
        dst[i] = PrecisionUtils::f16tof32(_src[i]) * scale + bias;
    }
}

void PrecisionUtils::f32tof16Arrays(short *dst, const float *src, size_t nelem, float scale, float bias) {
    for (size_t i = 0; i < nelem; i++) {
        dst[i] = PrecisionUtils::f32tof16(src[i] * scale + bias);
    }
}

// Function to convert F32 into F16
// F32: exp_bias:127 SEEEEEEE EMMMMMMM MMMMMMMM MMMMMMMM.
// F16: exp_bias:15  SEEEEEMM MMMMMMMM
#define EXP_MASK_F32 0x7F800000U
#define EXP_MASK_F16     0x7C00U


// small helper function to represent uint32_t value as float32
inline float asfloat(uint32_t v) {
    return *reinterpret_cast<float *>(&v);
}

// Function to convert F32 into F16
float PrecisionUtils::f16tof32(ie_fp16 x) {
    // this is storage for output result
    uint32_t u = x;

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
    } else if ((x & EXP_MASK_F16) == 0) {  // check for zero and denormals. both are converted to zero
        u = s;
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
    return *reinterpret_cast<float *>(&u);
}

// This function convert f32 to f16 with rounding to nearest value to minimize error
// the denormal values are converted to 0.
ie_fp16 PrecisionUtils::f32tof16(float x) {
    // create minimal positive normal f16 value in f32 format
    // exp:-14,mantissa:0 -> 2^-14 * 1.0
    static float min16 = asfloat((127 - 14) << 23);

    // create maximal positive normal f16 value in f32 and f16 formats
    // exp:15,mantissa:11111 -> 2^15 * 1.(11111)
    static float max16 = asfloat(((127 + 15) << 23) | 0x007FE000);
    static uint32_t max16f16 = ((15 + 15) << 10) | 0x3FF;

    // define and declare variable for intermidiate and output result
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
            return s | (v.u >> (23 - 10));  // return INF f16
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

namespace InferenceEngine {
    template<>
    void copyToFloat<uint8_t>(float *dst, const InferenceEngine::Blob *src) {
        if (!dst) {
            return;
        }
        const InferenceEngine::TBlob<uint8_t> *t_blob = dynamic_cast<const InferenceEngine::TBlob<uint8_t> *>(src);
        if (t_blob == nullptr) {
            THROW_IE_EXCEPTION << "input type is " << src->precision() << " but input is not " << typeid(uint8_t).name();
        }

        const uint8_t *srcPtr = t_blob->readOnly();
        if (srcPtr == nullptr) {
            THROW_IE_EXCEPTION << "Input data was not allocated.";
        }
        size_t multiple_of = 0;
        #if defined(__SSE4_2__)
        const bool is_aligned = (size_t(dst)&0xf) == 0;
        multiple_of = t_blob->size() - t_blob->size()%4;
        for (size_t i = 0; i < multiple_of; i+=4) {
                // Load four uchar elements to lower part of the __m128i
                const __m128i four_elements_uchar = _mm_castps_si128(_mm_load_ss(reinterpret_cast<const float*>(srcPtr + i)));
                // Convert four uchar elements to int
                const __m128i four_elements_int = _mm_cvtepu8_epi32(four_elements_uchar);
                // Convert int to float and store the result
                const __m128 four_elements_float = _mm_cvtepi32_ps(four_elements_int);
                if (is_aligned)
                    _mm_stream_ps(dst+i,  four_elements_float);
                else
                    _mm_storeu_ps(dst+i, four_elements_float);
            }
        #endif
        for (size_t i = multiple_of; i < t_blob->size(); i++) {
            dst[i] = srcPtr[i];
        }
    }
}  // namespace InferenceEngine