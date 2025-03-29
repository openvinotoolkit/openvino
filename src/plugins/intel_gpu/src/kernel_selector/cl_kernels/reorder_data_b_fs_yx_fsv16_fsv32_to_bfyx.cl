// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/fetch_data.cl"

#define INPUT0_GET_TILED_INDEX(ORDER) INPUT0_GET_INDEX(ORDER)
#define OUTPUT_GET_TILED_INDEX(ORDER) OUTPUT_GET_INDEX(ORDER)

#define INPUTVTYPE CAT(INPUT0_TYPE, DEFAULT_TILE_SIZE)
#define INPUTVTYPE_HALF CAT(INPUT0_TYPE, TILE_SIZE)
#define OUTPUTVTYPE CAT(OUTPUT_TYPE, TILE_SIZE)
#define VSTORE CAT(vstore, TILE_SIZE)
#define AS_INPUTVTYPE CAT(as_, INPUTVTYPE)
#define TO_OUTPUTVTYPE CAT(convert_, OUTPUTVTYPE)

#define GET_GLOBAL_ID(IDX) ((uint)get_global_id(IDX))
#define GET_LOCAL_ID(IDX) ((uint)get_local_id(IDX))
#define GET_LOCAL_SIZE(IDX) ((uint)get_local_size(IDX))

REQD_SUB_GROUP_SIZE(DEFAULT_STRIDE)
KERNEL (reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
    )
{
    const uint sub_group_id = get_sub_group_id();
    const uint sub_group_local_id = get_sub_group_local_id();

#if INPUT0_DIMS == 4
    #if OUTPUT_DIMS > 4
        const uint z = 0;
        const uint w = 0;
    #endif
    const uint y = GET_GLOBAL_ID(1);
#elif INPUT0_DIMS == 5
    #if OUTPUT_DIMS > 5
        const uint w = 0;
    #endif
    const uint y = GET_GLOBAL_ID(1) % INPUT0_SIZE_Y;
    const uint z = GET_GLOBAL_ID(1) / INPUT0_SIZE_Y;
#else
#error reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx.cl: input format - not supported
#endif

    const uint x = GET_GLOBAL_ID(0) / FSV_ALIGNMENT * DEFAULT_TILE_SIZE;

    const uint fs = GET_GLOBAL_ID(2) % INPUT0_FEATURE_SLICE_NUM;
    const uint b = GET_GLOBAL_ID(2) / INPUT0_FEATURE_SLICE_NUM;
    const uint f = fs * FSV_ALIGNMENT + sub_group_local_id;

    //read
    const uint x_pitch = FSV_ALIGNMENT;
    const uint y_pitch = x_pitch * INPUT0_SIZE_X;
#if INPUT0_DIMS == 4
    const uint fs_pitch = y_pitch * INPUT0_SIZE_Y;
    const uint b_pitch = fs_pitch * INPUT0_FEATURE_SLICE_NUM;
    const uint input_idx_tile = (b * b_pitch) + (fs * fs_pitch) + (y * y_pitch) + (x * x_pitch);
#else
    const uint z_pitch = y_pitch * INPUT0_SIZE_Y;
    const uint fs_pitch = z_pitch * INPUT0_SIZE_Z;
    const uint b_pitch = fs_pitch * INPUT0_FEATURE_SLICE_NUM;
    const uint input_idx_tile = (b * b_pitch) + (fs * fs_pitch) + (z * z_pitch) + (y * y_pitch) + (x * x_pitch);
#endif


#if (TILE_SIZE == DEFAULT_TILE_SIZE)
    // write index
    const uint output_idx = OUTPUT_GET_TILED_INDEX(OUTPUT_TILED_ORDER);

    if (F_NO_REMAINDER_CONDITION
#ifdef F_REMAINDER_SIZE
        || (F_REMAINDER_CONDITION && ((f % FSV_ALIGNMENT) < F_REMAINDER_SIZE))
#endif
    ) {
        #ifdef X_REMAINDER_SIZE
            if (X_REMAINDER_CONDITION) {
                // read
                INPUTVTYPE read_data;
                for (int j = 0; j < X_REMAINDER_SIZE; ++j) {
                     read_data[j] = AS_INPUT0_TYPE(_sub_group_block_read((const __global uint*)(input) + input_idx_tile + j * DEFAULT_STRIDE));
                }
                // write
                for (int i = 0 ; i < X_REMAINDER_SIZE; i++) {
                    output[output_idx + i] = TO_OUTPUT_TYPE(read_data[i]);
                }
            } else {
                // read
                INPUTVTYPE read_data = AS_INPUTVTYPE(_sub_group_block_read8((const __global uint*)(input) + input_idx_tile));
                // write
                VSTORE(TO_OUTPUTVTYPE(read_data), 0, output + output_idx);
            }
        #else
            // read
            INPUTVTYPE read_data = AS_INPUTVTYPE(_sub_group_block_read8((const __global uint*)(input) + input_idx_tile));
            // write
            VSTORE(TO_OUTPUTVTYPE(read_data), 0, output + output_idx);
        #endif
    }
#else
    const uint sgid_remainder = sub_group_id % 2;

    // read
    const uint input_idx_final = input_idx_tile + sgid_remainder * (DEFAULT_STRIDE * DEFAULT_TILE_SIZE);
    INPUTVTYPE read_data = AS_INPUTVTYPE(_sub_group_block_read8((const __global uint*)(input) + input_idx_final));
    INPUTVTYPE_HALF read_half1 = {read_data[0], read_data[2], read_data[4], read_data[6]};
    INPUTVTYPE_HALF read_half2 = {read_data[1], read_data[3], read_data[5], read_data[7]};

    // write
    const uint output_idx = OUTPUT_GET_TILED_INDEX(OUTPUT_TILED_ORDER);
    const uint output_idx_final = output_idx + (sgid_remainder * TILE_SIZE);

    if (F_NO_REMAINDER_CONDITION
#ifdef F_REMAINDER_SIZE
        || (F_REMAINDER_CONDITION && ((f % FSV_ALIGNMENT) < F_REMAINDER_SIZE))
#endif
        ) {
        #ifdef X_REMAINDER_SIZE
            if (X_REMAINDER_CONDITION) {
                const int nloop = X_REMAINDER_SIZE - (TILE_SIZE * sgid_remainder);
                for (int i = 0 ; i < min(nloop, TILE_SIZE); i++) {
                    output[output_idx_final + i] = TO_OUTPUT_TYPE(read_half1[i]);
                    #ifdef F_REMAINDER_SIZE
                    if ((f + DEFAULT_STRIDE) < OUTPUT_FEATURE_NUM)
                    #endif
                    {
                        output[output_idx_final + i + (OUTPUT_FEATURE_PITCH * DEFAULT_STRIDE)] = TO_OUTPUT_TYPE(read_half2[i]);
                    }
                }
            } else {
                VSTORE(TO_OUTPUTVTYPE(read_half1), 0, output + output_idx_final);
                #ifdef F_REMAINDER_SIZE
                if ((f + DEFAULT_STRIDE) < OUTPUT_FEATURE_NUM)
                #endif
                {
                    VSTORE(TO_OUTPUTVTYPE(read_half2), 0, output + output_idx_final + (OUTPUT_FEATURE_PITCH * DEFAULT_STRIDE));
                }
            }
        #else
            VSTORE(TO_OUTPUTVTYPE(read_half1), 0, output + output_idx_final);
            #ifdef F_REMAINDER_SIZE
            if((f + DEFAULT_STRIDE) < OUTPUT_FEATURE_NUM)
            #endif
            {
                VSTORE(TO_OUTPUTVTYPE(read_half2), 0, output + output_idx_final + (OUTPUT_FEATURE_PITCH * DEFAULT_STRIDE));
            }
        #endif
    }
#endif
}

#undef GET_LOCAL_SIZE
#undef GET_LOCAL_ID
#undef GET_GLOBAL_ID

#undef TO_OUTPUTVTYPE
#undef AS_INPUTVTYPE
#undef VSTORE
#undef OUTPUTVTYPE
#undef INPUTVTYPE_HALF
#undef INPUTVTYPE

#undef OUTPUT_GET_TILED_INDEX
#undef INPUT0_GET_TILED_INDEX
