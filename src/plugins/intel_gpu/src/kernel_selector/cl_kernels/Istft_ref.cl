// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// alternative: https://github.com/OpenCL/ComplexMath/blob/master/clcomplex.h
typedef float2 cfloat;
#define real(a)      ((a).s0)
#define imag(a)      ((a).s1)
#define crmult(a, b) ((cfloat)(real(a) * (b), imag(a) * (b)))
#define cadd(a, b)   ((cfloat)(real(a) + real(b), imag(a) + imag(b)))
#define csub(a, b)   ((cfloat)(real(a) - real(b), imag(a) - imag(b)))
#define expmi(x)     ((cfloat)(cos(x), -sin(x)))
#define czero()      ((cfloat)(0))

// Unoptimized, the most obvious stft impl from the definition.
KERNEL(istft_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* restrict signal, 
    const __global INPUT1_TYPE* restrict window,
    const __global INPUT2_TYPE* restrict frame_size_buff,
    const __global INPUT3_TYPE* restrict frame_step_buff,
    __global OUTPUT_TYPE* restrict output)
{

}