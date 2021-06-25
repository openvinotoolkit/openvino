// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "float16.h"
#include "include/math_utils.h"

FLOAT16::operator double() const {
    double d = (double)float16_to_float32(v);
    return d;
}

FLOAT16::operator float() const {
    float f = float16_to_float32(v);
    return f;
}

FLOAT16::FLOAT16(float f) { v = float32_to_float16(f); }
FLOAT16::FLOAT16(size_t s) { v = float32_to_float16(float(s)); }
FLOAT16::FLOAT16(int i) { v = float32_to_float16(float(i)); }
