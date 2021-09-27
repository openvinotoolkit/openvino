// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/data_types.cl"
#include "include/fetch_data.cl"

///////////////////////// Input Index /////////////////////////
inline uint FUNC(get_input_index)(uint b, uint f, uint w, uint z, uint y, uint x)
{
#if   INPUT0_SIMPLE && INPUT0_DIMS <= 4
    return GET_DATA_INDEX(INPUT0, b, f, y, x);
#elif INPUT0_SIMPLE && INPUT0_DIMS == 5
    return GET_DATA_INDEX_5D(INPUT0, b, f, z, y, x);
#elif INPUT0_SIMPLE && INPUT0_DIMS == 6
    return GET_DATA_INDEX_6D(INPUT0, b, f, w, z, y, x);
#elif INPUT0_LAYOUT_B_FS_ZYX_FSV16
    return GET_DATA_B_FS_ZYX_FSV16_INDEX(INPUT0, b, f, z, y, x);
#elif INPUT0_LAYOUT_BS_FS_ZYX_BSV16_FSV16
    return GET_DATA_BS_FS_ZYX_BSV16_FSV16_INDEX(INPUT0, b, f, z, y, x);
#elif INPUT0_LAYOUT_BS_FS_YX_BSV16_FSV16
    return GET_DATA_BS_FS_YX_BSV16_FSV16_INDEX(INPUT0, b, f, y, x);
#elif INPUT0_LAYOUT_BS_FS_YX_BSV32_FSV32
    return GET_DATA_BS_FS_YX_BSV32_FSV32_INDEX(INPUT0, b, f, y, x);
#else
#error concatenation_gpu_simple_ref.cl: input format - not supported
#endif
}

///////////////////////// Output Index /////////////////////////
inline uint FUNC(get_output_index)(uint b, uint f, uint w, uint z, uint y, uint x)
{
#if   OUTPUT_SIMPLE && OUTPUT_DIMS <= 4
    return GET_DATA_INDEX(OUTPUT, b, f, y, x);
#elif OUTPUT_SIMPLE && OUTPUT_DIMS == 5
    return GET_DATA_INDEX_5D(OUTPUT, b, f, z, y, x);
#elif OUTPUT_SIMPLE && OUTPUT_DIMS == 6
    return GET_DATA_INDEX_6D(OUTPUT, b, f, w, z, y, x);
#elif OUTPUT_LAYOUT_B_FS_ZYX_FSV16
    return GET_DATA_B_FS_ZYX_FSV16_INDEX(OUTPUT, b, f, z, y, x);
#elif OUTPUT_LAYOUT_BS_FS_ZYX_BSV16_FSV16
    return GET_DATA_BS_FS_ZYX_BSV16_FSV16_INDEX(OUTPUT, b, f, z, y, x);
#elif OUTPUT_LAYOUT_BS_FS_YX_BSV16_FSV16
    return GET_DATA_BS_FS_YX_BSV16_FSV16_INDEX(OUTPUT, b, f, y, x);
#elif OUTPUT_LAYOUT_BS_FS_YX_BSV32_FSV32
    return GET_DATA_BS_FS_YX_BSV32_FSV32_INDEX(OUTPUT, b, f, y, x);
#else
#error concatenation_gpu_simple_ref.cl: output format - not supported
#endif
}


KERNEL (concatenation_gpu_ref)(__global INPUT0_TYPE* input, __global OUTPUT_TYPE* output, uint output_offset_in_concat_axis)
{
    const uint x = (uint)get_global_id(0) % INPUT0_SIZE_X;
    const uint y = (uint)get_global_id(0) / INPUT0_SIZE_X;
    const uint z = (uint)get_global_id(1) % INPUT0_SIZE_Z;
    const uint w = (uint)get_global_id(1) / INPUT0_SIZE_Z;
    const uint f = (uint)get_global_id(2) % INPUT0_FEATURE_NUM;
    const uint b = (uint)get_global_id(2) / INPUT0_FEATURE_NUM;

    uint out_x = x;
    uint out_y = y;
    uint out_z = z;
    uint out_w = w;
    uint out_f = f;
    uint out_b = b;

#if CONCAT_X
    out_x += output_offset_in_concat_axis;
#elif CONCAT_Y
    out_y += output_offset_in_concat_axis;
#elif CONCAT_Z
    out_z += output_offset_in_concat_axis;
#elif CONCAT_W
    out_w += output_offset_in_concat_axis;
#elif CONCAT_FEATURE
    out_f += output_offset_in_concat_axis;
#elif CONCAT_BATCH
    out_b += output_offset_in_concat_axis;
#else
#   error concatenation_gpu_bfzyx_ref.cl: Unrecognized concat axis.
#endif

    uint input_offset  = FUNC_CALL(get_input_index)(b, f, w, z, y, x);
    uint output_offset = FUNC_CALL(get_output_index)(out_b, out_f, out_w, out_z, out_y, out_x);

    output[output_offset] = TO_OUTPUT_TYPE(ACTIVATION(input[input_offset], ACTIVATION_PARAMS));
}
