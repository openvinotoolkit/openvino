// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

inline int imad_SW(int acc, uchar4 input, char4 weight) __attribute__((overloadable)) {
    acc += input[0] * weight[0];
    acc += input[1] * weight[1];
    acc += input[2] * weight[2];
    acc += input[3] * weight[3];
    return acc;
}

inline int imad_SW(int acc, char4 input, char4 weight) __attribute__((overloadable)) {
    acc += input[0] * weight[0];
    acc += input[1] * weight[1];
    acc += input[2] * weight[2];
    acc += input[3] * weight[3];
    return acc;
}

inline int imad_SW(int acc, char4 input, uchar4 weight) __attribute__((overloadable)) {
    acc += input[0] * weight[0];
    acc += input[1] * weight[1];
    acc += input[2] * weight[2];
    acc += input[3] * weight[3];
    return acc;
}

inline int imad_SW(int acc, uchar4 input, uchar4 weight) __attribute__((overloadable)) {
    acc += input[0] * weight[0];
    acc += input[1] * weight[1];
    acc += input[2] * weight[2];
    acc += input[3] * weight[3];
    return acc;
}


#define IMAD(_O, _I, _W) imad_SW(_O, _I, _W)
