// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

KERNEL(stft_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* restrict signal, 
    const __global INPUT1_TYPE* restrict window,
    const __global INPUT1_TYPE* restrict frame_size,
    const __global INPUT1_TYPE* restrict frame_step,
    __global OUTPUT_TYPE* restrict output)
{

}