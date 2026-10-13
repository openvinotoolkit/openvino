// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_weights.cl"

KERNEL(reorder_weights_int4)(const __global INPUT0_TYPE* input, __global OUTPUT_TYPE* output) {
#if defined(INPUT0_LAYOUT_OIYX) && defined(OUTPUT_LAYOUT_OIYX)
#if INT2_PACKED
    // 2-bit packed weights padding: gather 4 values per output byte, zero-fill the padded area
    const uint out_byte_offset = get_global_id(0);
    OUTPUT_TYPE out = 0x0;
    unroll_for (uint j = 0; j < 4; ++j) {
        const uint output_index = out_byte_offset * 4 + j;
        const uint w = output_index % OUTPUT_INNERMOST_NUM;
        const uint h = output_index / OUTPUT_INNERMOST_NUM;
        if (w < INPUT0_INNERMOST_NUM) {
            const uint in_elem_offset = h * INPUT0_INNERMOST_NUM + w;
            out |= ((input[in_elem_offset / 4] >> ((in_elem_offset % 4) * 2)) & 0x3) << (j * 2);
        }
    }
    output[out_byte_offset] = out;
#else
    const uint out_byte_offset = get_global_id(0);
    const uint output_index = out_byte_offset * 2;
    const uint w = output_index % OUTPUT_INNERMOST_NUM;
    const uint h = output_index / OUTPUT_INNERMOST_NUM;
    const uint in_byte_offset = (h * INPUT0_INNERMOST_NUM + w) / 2;
    const bool within_pitch = (w + 1 < INPUT0_INNERMOST_NUM);

    if (h % 2 == 0) {
        if (within_pitch) {
            output[out_byte_offset] = input[in_byte_offset];
        } else {
            INPUT0_TYPE out0 = input[in_byte_offset] & 0x0F;
            output[out_byte_offset] = out0;
        }
    } else {
        INPUT0_TYPE out1 = (input[in_byte_offset] & 0xF0) >> 4;
        INPUT0_TYPE out0 = 0x0;
        if (within_pitch) {
            out0 = input[in_byte_offset + 1] & 0x0F;
        }
        output[out_byte_offset] = (out0 << 4) | out1;
    }
#endif
#elif defined(INPUT0_LAYOUT_IOYX) && defined(OUTPUT_LAYOUT_OIYX)
#if INT2_PACKED
    // 2-bit packed weights transpose: gather 4 values per output byte
    const uint out_byte_offset = get_global_id(0);
    OUTPUT_TYPE out = 0x0;
    unroll_for (uint j = 0; j < 4; ++j) {
        const uint offset = out_byte_offset * 4 + j;
        const uint i = offset % OUTPUT_IFM_NUM;
        const uint o = offset / OUTPUT_IFM_NUM;
        const uint input_offset = GET_FILTER_INDEX(INPUT0, 0, o, i, 0, 0);
        out |= ((input[input_offset / 4] >> ((input_offset % 4) * 2)) & 0x3) << (j * 2);
    }
    output[out_byte_offset] = out;
#else
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
#endif
#elif defined(OUTPUT_LAYOUT_OS_IYX_OSV16)
#if INT2_PACKED
    // osv16 layout for 2-bit packed weight (4 values per byte along IFM)
    // f0_k0k1k2k3 | f1_k0k1k2k3 | ....  | f15_k0k1k2k3
    // f0_k4k5k6k7 | f1_k4k5k6k7 | ....  | f15_k4k5k6k7
    // ...
    const unsigned o = (uint)get_global_id(0);
    const unsigned i = (uint)get_global_id(1) * 4;

    const uint input0_offset = GET_FILTER_INDEX(INPUT0, 0, o, i, 0, 0);

    INPUT0_TYPE packed_out_channels = input[input0_offset / 4] & 0xFF;

    const uint output_idx = GET_FILTER_OS_IYX_OSV_INDEX_INT2_PACKED(OUTPUT, o, i/4, 0, 0, 16); // Calculate offset as osv16 due to packing
    output[output_idx] = packed_out_channels;
#else
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
#endif
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
#if INT2_PACKED
    // os_is_yx_osv64_isv2 layout for 2-bit packed weight (4 values per byte):
    // a 64-OFM block is a sequence of 32-byte k-pair lines; the byte at position p of line kb
    // packs k[2*kb] and k[2*kb+1] of two output features q and q + 16 (LSB-first:
    // k[2*kb] of q, k[2*kb+1] of q, k[2*kb] of q+16, k[2*kb+1] of q+16), where q = p % 16 + 32 * (p / 16):
    // f0_k0k1 f16_k0k1 | f1_k0k1 f17_k0k1 | ... | f15_k0k1 f31_k0k1 || f32_k0k1 f48_k0k1 | ... | f47_k0k1 f63_k0k1
    // f0_k2k3 f16_k2k3 | f1_k2k3 f17_k2k3 | ... | f15_k2k3 f31_k2k3 || f32_k2k3 f48_k2k3 | ... | f47_k2k3 f63_k2k3
    // ...
    // This matches the bf_tiled fetch windows: for TILE_OFM 4 / TILE_K 2 a lane reads the bytes at
    // positions sglid and sglid + 16 of each line; for the SLM kernel (TILE_OFM 2 / TILE_K 4) a lane
    // reads position sglid of two consecutive lines.
    const unsigned o = (uint)get_global_id(0);   // 64-OFM block * 32 + byte position in the line
    const unsigned kb = (uint)get_global_id(1);  // k-pair index

    const unsigned p = o % 32;
    const unsigned q = (p % 16) + (p / 16) * 32;
    const unsigned o0 = (o / 32) * 64 + q;
    const unsigned o1 = o0 + 16;
    const unsigned k = kb * 2;

    const uint input0_offset = GET_FILTER_INDEX(INPUT0, 0, o0, k, 0, 0);
    const uint input1_offset = GET_FILTER_INDEX(INPUT0, 0, o1, k, 0, 0);

    // k is even and IFM is aligned to 4 (values per byte), so a k-pair never straddles
    // an input byte boundary; extract the 2 u2 values (4 bits) of each output feature
    INPUT0_TYPE in0 = (input[input0_offset / 4] >> ((input0_offset % 4) * 2)) & 0x0F;
    INPUT0_TYPE in1 = (input[input1_offset / 4] >> ((input1_offset % 4) * 2)) & 0x0F;

    INPUT0_TYPE packed_out_channels = in0 | (in1 << 4);

    const uint output_idx = GET_FILTER_OS_IS_YX_OSV_ISV_INDEX_INT2_PACKED(OUTPUT, o, kb, 0, 0, 32); // 32 bytes per k-pair line
    output[output_idx] = packed_out_channels;
#else
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
#endif
#else
#error "reorder_weights_int4: unsupported layouts combination"
#endif
}
