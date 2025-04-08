// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_weights.cl"

KERNEL(reorder_weights_int4)(const __global INPUT0_TYPE* input, __global OUTPUT_TYPE* output) {
#if defined(INPUT0_LAYOUT_IOYX) && defined(OUTPUT_LAYOUT_OIYX)
    const uint out_byte_offset = get_global_id(0);

    const uint offset0 = out_byte_offset * 2 + 0;
    const uint offset1 = out_byte_offset * 2 + 1;

    const uint i0 = offset0 % OUTPUT_IFM_NUM;
    const uint i1 = offset1 % OUTPUT_IFM_NUM;

    const uint o0 = offset0 / OUTPUT_IFM_NUM;
    const uint o1 = offset1 / OUTPUT_IFM_NUM;

    const uint input0_offset = GET_FILTER_INDEX(INPUT0, 0, o0, i0, 0, 0);
    const uint input1_offset = GET_FILTER_INDEX(INPUT0, 0, o1, i1, 0, 0);

    const uint input0_idx = input0_offset % 2;
    const uint input1_idx = input1_offset % 2;

    INPUT0_TYPE in0 = (input[input0_offset / 2] >> input0_idx*4) & 0x0F;
    INPUT0_TYPE in1 = (input[input1_offset / 2] >> input1_idx*4) & 0x0F;

    OUTPUT_TYPE out = in0 | (in1 << 4);
    output[out_byte_offset] = out;
#elif defined(OUTPUT_LAYOUT_OS_IYX_OSV16)
    // osv32_isv2 layout for int4 packed weight
    // f0_k0k1 | f1_k0k1 | ....  | f15_k0k1
    // f0_k2k3 | f1_k2k3 | ....  | f15_k2k3
    // f0_k3k4 | f1_k3k4 | ....  | f15_k3k4
    // ...
    // f0_k(K/2-2)k(K/2-1) | f1_k(K/2-2)k(K/2-1) | ....f15_k(K/2-2)k(K/2-1)
    // -------------------------------------
    // f16_k2k3 | f17_k2k3 | ... | f31_k2k3
    // ...
    const unsigned o = (uint)get_global_id(0);
    const unsigned i = (uint)get_global_id(1) * 2;

    const uint input0_offset = GET_FILTER_INDEX(INPUT0, 0, o, i, 0, 0);

    INPUT0_TYPE in1 = input[input0_offset / 2] & 0xFF;

    INPUT0_TYPE packed_out_channels = in1;

    const uint output_idx = GET_FILTER_OS_IYX_OSV_INDEX_INT4_PACKED(OUTPUT, o, i/2, 0, 0, 16); // Calculate offset as osv16 due to packing
    output[output_idx] = packed_out_channels;
#elif defined(OUTPUT_LAYOUT_OS_IYX_OSV32)
    // os_iyx osv32 layout for int4 packed weight
    // k0_f0f16 | k0_f1f17 | .... | k0_f15f31 || k1_f0f16 | k1_f1f17 | ... | k1_f15f31
    // k2_f0f16 | k2_f1f17 | .... | k2_f15f31 || k3_f0f16 | k3_f1f17 | ... | k3_f15f31
    // ...
    const unsigned o = (uint)get_global_id(0);
    const unsigned i = (uint)get_global_id(1);

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
#elif defined(OUTPUT_LAYOUT_OS_IS_YX_OSV32_ISV2)
    // osv32_isv2 layout for int4 packed weight
    // f0_k0k1 | f1_k0k1 | ....  | f15_k0k1|| f16_k0k1 | f17_k0k1 | ... | f31_k0k1
    // f0_k2k3 | f1_k2k3 | ....  | f15_k2k3|| f16_k2k3 | f17_k2k3 | ... | f31_k2k3
    // ...
    const unsigned o = (uint)get_global_id(0);
    const unsigned i = (uint)get_global_id(1) * 2;

    const uint input0_offset = GET_FILTER_INDEX(INPUT0, 0, o, i, 0, 0);

    INPUT0_TYPE in1 = input[input0_offset / 2] & 0xFF;

    INPUT0_TYPE packed_out_channels = in1;

    const uint output_idx = GET_FILTER_OS_IS_YX_OSV_ISV_INDEX_INT4_PACKED(OUTPUT, o, i/2, 0, 0, 32); // Calculate offset as osv16 due to packing
    output[output_idx] = packed_out_channels;
#elif defined(OUTPUT_LAYOUT_OS_IYX_OSV64)
    // os_iyx_osv64 layout for int4 packed weight
    // k0_f0f16 | k0_f1f17 | .... | k0_f15f31 || k0_f32f48 | k0_f33f49 | .... | k0_f47f63 || k1_f0f16 | k1_f1f17 | .... | k1_f15f31 || k1_f32f48 | k1_f33f49 | .... | k1_f47f63 ||
    // k2_f0f16 | k2_f1f17 | .... | k2_f15f31 || k2_f32f48 | k2_f33f49 | .... | k2_f47f63 || k3_f0f16 | k3_f1f17 | .... | k3_f15f31 || k3_f32f48 | k3_f33f49 | .... | k3_f47f63 ||
    // ...
    const unsigned o = (uint)get_global_id(0);
    const unsigned i = (uint)get_global_id(1);

    // Calculate offsets for 2 contiguous values in the 8-bit packed format
    const unsigned o0 = (o / 16) * 32 + (o % 16);
    const unsigned o1 = (o / 16) * 32 + (o % 16) + 16;

    // Calculate the input buffer offests
    const uint input0_offset = GET_FILTER_INDEX(INPUT0, 0, o0, i, 0, 0);
    const uint input1_offset = GET_FILTER_INDEX(INPUT0, 0, o1, i, 0, 0);

    // Determine the bit position within each 8-bit value
    const uint input0_idx = input0_offset % 2;
    const uint input1_idx = input1_offset % 2;

    // Extract 4-bit values from the input buffer
    INPUT0_TYPE in0 = (input[input0_offset / 2] >> input0_idx*4) & 0x0F;
    INPUT0_TYPE in1 = (input[input1_offset / 2] >> input1_idx*4) & 0x0F;

    // Combine the 4-bit values into a single 8-bit value
    INPUT0_TYPE packed_out_channels = in0 | (in1 << 4);

    // Calculate the output buffer index for the packed 8-bit data
    const uint output_idx = GET_FILTER_OS_IYX_OSV_INDEX(OUTPUT, o, i, 0, 0, 64 / 2);
    output[output_idx] = packed_out_channels;
#elif defined(OUTPUT_LAYOUT_OS_IS_YX_OSV64_ISV2)
    // os_is_yx_osv64_isv2 layout for int4 packed weight
    // f0_k0k1 | f1_k0k1 | .... | f15_k0k1 || f16_k0k1 | f17_k0k1 | .... | f31_k0k1 || f32_k0k1 | f33_k0k1 | .... | kf47_k0k1 || f48_k0k1 | f49_k0k1 | .... | f63_k0k1 ||
    // f0_k2k3 | f1_k2k3 | .... | f15_k2k3 || f16_k2k3 | f17_k2k3 | .... | f31_k2k3 || f32_k2k3 | f33_k2k3 | .... | kf47_k2k3 || f48_k2k3 | f49_k2k3 | .... | f63_k2k3 ||
    // ...
    const unsigned o = (uint)get_global_id(0);
    const unsigned i = (uint)get_global_id(1) * 2;

    // Calculate the input buffer offset
    const uint input0_offset = GET_FILTER_INDEX(INPUT0, 0, o, i, 0, 0);

    // Extract 8-bit packed value from the input buffer
    INPUT0_TYPE in1 = input[input0_offset / 2] & 0xFF;

    // Prepare the output value by directly using the extracted value
    // Since the data is packed, no further processing is needed here
    INPUT0_TYPE packed_out_channels = in1;

    // Calculate the output buffer index for the packed 8-bit data
    const uint output_idx = GET_FILTER_OS_IS_YX_OSV_ISV_INDEX_INT4_PACKED(OUTPUT, o, i/2, 0, 0, 64);
    output[output_idx] = packed_out_channels;
#else
#error "reorder_weights_int4: unsupported layouts combination"
#endif
}
