// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#if OUTPUT_DIMS != 4
#error "dynamic_quantize_gpu_ref.cl: Unsupported output dimension"
#endif

KERNEL(dynamic_quantize_gpu_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    __global OUTPUT1_TYPE* output_scale)
{
    const uint bf = (uint)get_global_id(0);
    const uint b = (uint)get_global_id(0) / INPUT0_FEATURE_NUM;
    const uint f = (uint)get_global_id(0) % INPUT0_FEATURE_NUM;
    const uint y = (uint)get_global_id(1);
    const uint scale_idx = OUTPUT1_GET_INDEX(b, f, y, 0);

    half max_val = 0.0001h;
    for (int y_off = 0; y_off < (get_global_size(1) == 1 ? INPUT0_SIZE_Y : 1); y_off++) {
        const uint offset = INPUT0_GET_INDEX(b, f, y + y_off, 0);
        int x;
        for (x = 0; x < INPUT0_SIZE_X / 8; x++) {
            half8 val = as_half8(vload8(0, (ushort*)input + offset + x * 8));
            half8 abs_val = fabs(val);

            for (int j = 0; j < 8; j++)
                max_val = fmax(max_val, abs_val[j]);
        }
        x *= 8;
        for (; x < INPUT0_SIZE_X; x++)
            max_val = fmax(max_val, fabs(input[offset + x]));
    }

    half scale = 127.0h / max_val;
    for (int y_off = 0; y_off < (get_global_size(1) == 1 ? INPUT0_SIZE_Y : 1); y_off++) {
        const uint in_offset = INPUT0_GET_INDEX(b, f, y + y_off, 0);
        const uint out_offset = OUTPUT_GET_INDEX(b, f, y + y_off, 0);
        int x;
        for (x = 0; x < INPUT0_SIZE_X / 8; x++) {
            half8 val = as_half8(vload8(0, (ushort*)input + in_offset + x * 8));
            val *= scale;
            vstore8(convert_char8(val), 0, output + out_offset + x * 8);
        }
        x *= 8;
        for (; x < INPUT0_SIZE_X; x++)
            output[out_offset + x] = convert_char(input[in_offset + x] * scale);
    }

    output_scale[scale_idx] = 1.0h / scale;
}
