// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// AllReduce sum kernels used by the TP_GPU plugin.
//
// All buffers are device-local on the executing rank's GPU; peer
// contributions are staged via explicit cross-device USM copies before
// the kernel is launched (see TPDeviceCoordinator).

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void allreduce_sum_f16(__global half* restrict dst,
                                __global const half* restrict src0,
                                __global const half* restrict src1,
                                ulong n) {
    ulong i = get_global_id(0);
    if (i < n) {
        dst[i] = src0[i] + src1[i];
    }
}

__kernel void allreduce_sum_f32(__global float* restrict dst,
                                __global const float* restrict src0,
                                __global const float* restrict src1,
                                ulong n) {
    ulong i = get_global_id(0);
    if (i < n) {
        dst[i] = src0[i] + src1[i];
    }
}
