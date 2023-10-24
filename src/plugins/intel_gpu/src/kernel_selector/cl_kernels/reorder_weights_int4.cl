// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_weights.cl"

KERNEL(reorder_weights_int4)(const __global INPUT0_TYPE* input, __global OUTPUT_TYPE* output) {
    const unsigned o = (uint)get_global_id(0);
    const unsigned i = (uint)get_global_id(1);

#if defined(INPUT0_LAYOUT_IOYX) && defined(OUTPUT_LAYOUT_OIYX)
    const uint output_offset = (o*OUTPUT_IFM_NUM + 2*i);

    const uint input0_offset = o + (i*2+0)*INPUT0_OFM_NUM;
    const uint input1_offset = o + (i*2+1)*INPUT0_OFM_NUM;

    const uint input0_idx = input0_offset % 2;
    const uint input1_idx = input1_offset % 2;

    INPUT0_TYPE in0 = (input[input0_offset / 2] >> input0_idx*4) & 0x0F;
    INPUT0_TYPE in1 = (input[input1_offset / 2] >> input1_idx*4) & 0x0F;

    INPUT0_TYPE packed_out_channels = in0 | (in1 << 4);
    output[output_offset / 2] = packed_out_channels;
#elif defined(OUTPUT_LAYOUT_OS_IYX_OSV32)
    const unsigned o0 = (o / 16) * 32 + (o % 16);
    const unsigned o1 = (o / 16) * 32 + (o % 16) + 16;

    const uint input0_offset = GET_FILTER_INDEX(INPUT0, 0, o0, i, 0, 0);
    const uint input1_offset = GET_FILTER_INDEX(INPUT0, 0, o1, i, 0, 0);

    const uint input0_idx = input0_offset % 2;
    const uint input1_idx = input1_offset % 2;

    INPUT0_TYPE in0 = (input[input0_offset / 2] >> input0_idx*4) & 0x0F;
    INPUT0_TYPE in1 = (input[input1_offset / 2] >> input1_idx*4) & 0x0F;

    INPUT0_TYPE packed_out_channels = in0 | (in1 << 4);

    const uint output_idx = GET_FILTER_OS_IYX_OSV_INDEX(OUTPUT, o, i, 0, 0, 32 / 2); // Calculate offset as osv16 due to packing
    output[output_idx] = packed_out_channels;
#else
#error "reorder_weights_int4: unsupported layouts combination"
#endif
}
