// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

typedef short ie_fp16;

// Function to convert F32 into F16
// F32: exp_bias:127 SEEEEEEE EMMMMMMM MMMMMMMM MMMMMMMM.
// F16: exp_bias:15  SEEEEEMM MMMMMMMM
#define EXP_MASK_F32 0x7F800000U
#define EXP_MASK_F16     0x7C00U

// small helper function to represent uint32_t value as float32
inline float asfloat(uint32_t v) {
    // Both type-punning casts and unions are UB per C++ spec
    // But compilers usually only break code with casts
    union {
        float f;
        uint32_t i;
    } u;
    u.i = v;
    return u.f;
}

// Function to convert F32 into F16
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

