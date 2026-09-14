// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SELECTIVE_SSM_TYPE_UTILS
#define SELECTIVE_SSM_TYPE_UTILS

#include "include/batch_headers/bf16_utils.cl"

inline float ssm_to_float(float value) __attribute__((overloadable)) {
    return value;
}

inline float ssm_to_float(half value) __attribute__((overloadable)) {
    return convert_float(value);
}

inline float ssm_to_float(ushort value) __attribute__((overloadable)) {
    return _convert_as_bfloat16_float(value);
}

#endif
