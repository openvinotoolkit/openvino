// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_weights.cl"

KERNEL(reorder_weights_int4)(const __global INPUT0_TYPE* input, __global OUTPUT_TYPE* output) {
    const unsigned o = (uint)get_global_id(0);
    const unsigned i = (uint)get_global_id(1);

#if defined(INPUT0_LAYOUT_IOYX) && defined(OUTPUT_LAYOUT_OIYX)
    // load 2 adjucent input channels to store full byte
    const uint output_offset = (o*OUTPUT_IFM_NUM + 2*i);

    const uint input0_offset = o + (i*2+0)*INPUT0_OFM_NUM;
    const uint input1_offset = o + (i*2+1)*INPUT0_OFM_NUM;

    const uint input0_idx = (input0_offset + 1) % 2; // INVERTED! Should be input0_offset % 2
    const uint input1_idx = (input1_offset + 1) % 2; // INVERTED! Should be input1_offset % 2

    INPUT0_TYPE in0 = (input[input0_offset / 2] >> input0_idx*4) & 0x0F;
    INPUT0_TYPE in1 = (input[input1_offset / 2] >> input1_idx*4) & 0x0F;

    INPUT0_TYPE packed_out_channels = in1 | (in0 << 4); // INVERTED! Should be in0 | (in1 << 4)
    output[output_offset / 2] = packed_out_channels;
#elif defined(OUTPUT_LAYOUT_OS_IYX_OSV32)
    const unsigned o0 = (o / 16) * 32 + (o % 16);
    const unsigned o1 = (o / 16) * 32 + (o % 16) + 16;

#if defined(INPUT0_LAYOUT_OIYX)
    const uint input0_offset = o0*INPUT0_IFM_NUM + i;
    const uint input1_offset = o1*INPUT0_IFM_NUM + i;
#elif defined(INPUT0_LAYOUT_IOYX)
    const uint input0_offset = o0 + i*INPUT0_OFM_NUM;
    const uint input1_offset = o1 + i*INPUT0_OFM_NUM;
#endif

    const uint input0_idx = (input0_offset + 1) % 2; // INVERTED! Should be input0_offset % 2
    const uint input1_idx = (input1_offset + 1) % 2; // INVERTED! Should be input1_offset % 2

    INPUT0_TYPE in0 = (input[input0_offset / 2] >> input0_idx*4) & 0x0F;
    INPUT0_TYPE in1 = (input[input1_offset / 2] >> input1_idx*4) & 0x0F;

    INPUT0_TYPE packed_out_channels = in1 | (in0 << 4); // INVERTED! Should be in0 | (in1 << 4)

    const uint osv_size = 32;
    const uint osv_byte_size = osv_size / 2;
    const uint i_offset = osv_byte_size;
    const uint os_offset = i_offset * OUTPUT_IFM_NUM;
    const uint os_idx = o / osv_byte_size;
    const uint ov_idx = o % osv_byte_size;

    uint output_idx = os_idx * os_offset + i * i_offset + ov_idx;

    output[output_idx] = packed_out_channels;
#else
#error "reorder_weights_int4: unsupported layouts combination"
#endif
}
