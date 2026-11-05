// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

// Packed UINT2 fast path for bfyx order {1, 0, 2, 3}: [B, F, Y, X] -> [F, B, Y, X].
// Requires static, contiguous input/output with zero offsets and no fused operations.
// Y * X must be divisible by 64 (16 packed bytes). Each work-item copies 16 bytes
// from plane (b, f) to (f, b), preserving the spatial order without unpacking UINT2 values.
KERNEL(permute_bf_swap)(const __global INPUT0_TYPE* input, __global OUTPUT_TYPE* output) {
    const uint block = (uint)get_global_id(0);
    const uint f = (uint)get_global_id(1);
    const uint b = (uint)get_global_id(2);

    const uint input_block = (b * INPUT0_FEATURE_NUM + f) * PLANE_BLOCKS + block;
    const uint output_block = (f * INPUT0_BATCH_NUM + b) * PLANE_BLOCKS + block;

    const uchar16 values = vload16(input_block, (const __global uchar*)input);
    vstore16(values, output_block, (__global uchar*)output);
}
