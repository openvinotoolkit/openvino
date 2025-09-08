// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Contains logic derived from TensorFlow’s bfloat16 implementation
// https://github.com/tensorflow/tensorflow/blob/d354efc/tensorflow/core/lib/float16/float16.h
// Copyright notice from original source file is as follows.

//*******************************************************************************
//  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//    http://www.apache.org/licenses/LICENSE-2.0
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//==============================================================================

#include "openvino/core/type/float16.hpp"

#include <cmath>
#include <iostream>
#include <limits>

using namespace ov;

static_assert(sizeof(float16) == 2, "class float16 must be exactly 2 bytes");

float16::float16(float value) {
    // Work in 32-bit and shift right 16 in the end
    union {
        float fv;
        uint32_t iv;
    };
    fv = value;
    // sign
    constexpr uint32_t smask = 0x80000000;
    // floqt32 exp
    constexpr uint32_t emask_32 = 0x7F800000;
    // float32 frac
    constexpr uint32_t fmask_32 = 0x007fffff;
    // float16 exp
    constexpr uint32_t emask_16 = 0x7c000000;
    // float16 frac
    constexpr uint32_t fmask_16 = 0x03ff0000;
    // bits for half to even round
    constexpr uint32_t rhalf_16 = 0x0001ffff;
    // bit value for normal round
    constexpr uint32_t rnorm_16 = 0x00007fff;
    // bit value for half to even round
    constexpr uint32_t reven_16 = 0x00008000;
    // value for an non-half to even round
    constexpr uint32_t rodd_16 = 0x000018000;

    // exp bits in position
    uint32_t biased_exp_field_32 = iv & emask_32;
    uint32_t frac = (iv & fmask_32) << 3;
    if (biased_exp_field_32 == emask_32) {
        // Inf or NaN
        if (frac != 0) {
            // NaN
            frac &= fmask_16;
            if (frac == 0) {
                frac = 0x00010000;
            }
        }
        m_value = ((iv & smask) | emask_16 | frac) >> 16;
        return;
    }
    if (biased_exp_field_32 == 0) {
        m_value = (iv & smask) >> 16;
        return;
    }
    int16_t biased_exp_16 = (biased_exp_field_32 >> 23) - 127 + 15;
    // In the normalized_16 realm
    if ((frac & rhalf_16) == rodd_16 || (frac & rnorm_16) != 0) {
        frac += reven_16;
        if (0 != (frac & emask_16)) {
            frac &= emask_16;
            biased_exp_16++;
        }
    }
    frac &= fmask_16;
    if (biased_exp_16 > 30) {
        // Infinity
        m_value = ((iv & smask) | emask_16 | 0) >> 16;
        return;
    }
    if (biased_exp_16 > 0) {
        m_value = ((iv & smask) | biased_exp_16 << 26 | frac) >> 16;
        return;
    }
    // Restore the hidden 1
    frac = 0x04000000 | ((iv & fmask_32) << 3);
    // Will any bits be shifted off?
    int32_t shift = biased_exp_16 < -30 ? 0 : (1 << (1 - biased_exp_16));
    uint32_t sticky = (frac & (shift - 1)) ? 1 : 0;
    if (1 + (-biased_exp_16) > 31)
        frac = 0;
    else
        frac >>= 1 + (-biased_exp_16);
    frac |= sticky;
    if (((frac & rhalf_16) == rodd_16) || ((frac & rnorm_16) != 0)) {
        frac += reven_16;
    }
    m_value = ((iv & smask) | frac) >> 16;
}

std::string float16::to_string() const {
    return std::to_string(static_cast<float>(*this));
}

size_t float16::size() const {
    return sizeof(m_value);
}

float16::operator float() const {
    union {
        uint32_t i_val;
        float f_val;
    };
    uint32_t exp = 0x1F & (m_value >> frac_size);
    uint32_t fexp = exp + 127 - 15;
    uint32_t frac = m_value & 0x03FF;
    if (exp == 0) {
        if (frac == 0) {
            fexp = 0;
        } else {
            // Normalize
            fexp++;
            while (0 == (frac & 0x0400)) {
                fexp--;
                frac = frac << 1;
            }
            frac &= 0x03FF;
        }
    } else if (exp == 0x1F) {
        fexp = 0xFF;
    }
    frac = frac << (23 - frac_size);
    i_val = static_cast<uint32_t>((m_value & 0x8000)) << 16 | (fexp << 23) | frac;
    return f_val;
}

bool std::isnan(float16 x) {
    // Sign doesn't matter, frac not zero (infinity)
    return (x.to_bits() & 0x7FFF) > 0x7c00;
}

uint16_t float16::to_bits() const {
    return m_value;
}
