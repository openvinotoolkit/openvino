// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TP_VEC 8

__kernel void allreduce_sum_f16(__global half* restrict dst,
                                __global const half* restrict src0,
                                __global const half* restrict src1,
                                ulong n) {
    const ulong i = get_global_id(0);
    const ulong base = i * TP_VEC;
    if (base + TP_VEC <= n) {
        vstore8(vload8(i, src0) + vload8(i, src1), i, dst);
    } else if (base < n) {
        for (ulong k = base; k < n; ++k) {
            dst[k] = src0[k] + src1[k];
        }
    }
}

__kernel void allreduce_sum_f32(__global float* restrict dst,
                                __global const float* restrict src0,
                                __global const float* restrict src1,
                                ulong n) {
    const ulong i = get_global_id(0);
    const ulong base = i * TP_VEC;
    if (base + TP_VEC <= n) {
        vstore8(vload8(i, src0) + vload8(i, src1), i, dst);
    } else if (base < n) {
        for (ulong k = base; k < n; ++k) {
            dst[k] = src0[k] + src1[k];
        }
    }
}
