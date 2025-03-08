// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
#    include <immintrin.h>
#endif

#include "openvino/core/visibility.hpp"

namespace ov::intel_cpu {
static constexpr unsigned int FTZ_FLAG = 0x8000;
static constexpr unsigned int DAZ_FLAG = 0x0040;

bool flush_to_zero(bool on);
bool denormals_as_zero(bool on);

#ifdef OPENVINO_ARCH_X86_64

bool flush_to_zero(bool on) {
    unsigned int mxcsr = _mm_getcsr();
    if (on) {
        mxcsr |= FTZ_FLAG;
    } else {
        mxcsr &= ~FTZ_FLAG;
    }
    _mm_setcsr(mxcsr);
    return true;
}

bool denormals_as_zero(bool on) {
    unsigned int mxcsr = _mm_getcsr();
    if (on) {
        mxcsr |= DAZ_FLAG;
    } else {
        mxcsr &= ~DAZ_FLAG;
    }
    _mm_setcsr(mxcsr);
    return true;
}
#else  // OPENVINO_ARCH_X86_64
#    if defined(__SSE__) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
bool flush_to_zero(bool on) {
    unsigned int mxcsr = _mm_getcsr();
    if (on) {
        mxcsr |= FTZ_FLAG;
    } else {
        mxcsr &= ~FTZ_FLAG;
    }
    _mm_setcsr(mxcsr);
    return true;
}

bool denormals_as_zero(bool on) {
    // for some processor, DAZ flag is a reserved bit even SSE is available. Set 1 to this flag will generate #GP
    // exception.
    struct {
        char fcw0;
        char fcw1;
        char fsw0;
        char fsw1;
        char ftw;
        char rsvd;
        char fop0;
        char fop1;
        unsigned int fpu_ip;             // 8-11
        unsigned int cs_fpu_ip;          // 12-15
        unsigned int fpu_dp;             // 16-19
        unsigned int ds_fpu_dp;          // 20-23
        unsigned int mxcsr;              // 24-27
        unsigned int mxcsr_mask;         // 28-31
        unsigned int st_mm_status_0[8];  // 32 byte
        unsigned int st_mm_status_1[8];  // 32 byte
        unsigned int st_mm_status_2[8];  // 32 byte
        unsigned int st_mm_status_3[8];  // 32 byte
        unsigned int xmm_status_0[8];
        unsigned int xmm_status_1[8];
        unsigned int xmm_status_2[8];
        unsigned int xmm_status_3[8];
        unsigned int xmm_status_4[8];
        unsigned int xmm_status_5[8];
        unsigned int xmm_status_6[8];
        unsigned int xmm_status_7[8];  // 8 * 32 byte == 16 * 16 byte == 16 * 128 bit
        unsigned int padding_0[8];
        unsigned int padding_1[8];
        unsigned int padding_2[8];  // 3 * 32 byte
    } fxsave_area;                  // should be at least 16 byte aligned. fxsave_area is 32 byte aligned.

    fxsave_area.mxcsr_mask = 0;
#        ifdef _WIN32
    _fxsave(&fxsave_area);
#        else
    __builtin_ia32_fxsave(&fxsave_area);
#        endif
    unsigned int mxcsr_mask = fxsave_area.mxcsr_mask;  // 0 value for the bit indicate reserved

    unsigned int mxcsr = _mm_getcsr();
    if (on) {
        if (mxcsr_mask & DAZ_FLAG) {
            mxcsr |= DAZ_FLAG;
            _mm_setcsr(mxcsr);
            return true;
        } else {
            return false;
        }
    } else {
        mxcsr &= ~DAZ_FLAG;
        _mm_setcsr(mxcsr);
        return true;
    }
}
#    else
bool flush_to_zero(bool on) {
    return false;
}
bool denormals_as_zero(bool on) {
    return false;
}
#    endif
#endif  // OPENVINO_ARCH_X86_64

}  // namespace ov::intel_cpu
