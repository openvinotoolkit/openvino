// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// AllReduce sum: dst[i] = src0[i] + src1[i]
// Both buffers are local to the executing device (peer data is staged
// via explicit P2P copy before kernel launch).

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
