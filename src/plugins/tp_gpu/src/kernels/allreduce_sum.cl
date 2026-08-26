// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// AllReduce sum kernels used by the TP_GPU plugin.
//
// All buffers are device-local on the executing rank's GPU; peer
// contributions are staged via explicit cross-device USM copies before
// the kernel is launched (see TPDeviceCoordinator).

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Elements folded by one work item.  The reduction is purely memory bound --
// two loads and a store per element, no arithmetic worth the name -- so the
// only thing that matters is issuing wide enough accesses to saturate the
// memory pipe.  Eight elements is 16 bytes for f16 and 32 for f32.
//
// The vector path needs the three pointers to be naturally aligned.  They are:
// ring chunk boundaries are multiples of 128 elements (kRingAlignElems in the
// coordinator, chosen for exactly this reason) and every staging allocation is
// made with an alignment of at least 64 bytes.  Only the element count can be
// arbitrary, because the last chunk is clipped to the payload, so the work
// item that straddles the end falls back to scalar.
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
