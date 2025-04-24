// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

#ifndef SG_SIZE
    #define SG_SIZE   16
    #define SG_SIZE_NEEDSUNDEF_
#endif
#ifndef INB_ARRAY_SIZE
    #define INB_ARRAY_SIZE   8
    #define INB_ARRAY_SIZE_NEEDSUNDEF_
#endif
#ifndef UNIT_FILL_VAL
    #ifdef MAX_OUT
        #define UNIT_FILL_VAL   UNIT_VAL_MIN
    #else
        #define UNIT_FILL_VAL   UNIT_VAL_MAX
    #endif
    #define UNIT_FILL_VAL_NEEDSUNDEF_
#endif
#if MAX_OUT
    #define OP_ARG_REL   >
#else
    #define OP_ARG_REL   <
#endif


#if SG_SIZE != 8 && SG_SIZE != 16
    #error This kernel does not support specified sub-group size.
#endif
#if TOP_K > INB_ARRAY_SIZE * SG_SIZE || TOP_K <= 0
    #error This kernel does not support specified "TOP_K" JIT parameter.
#endif


REQD_SUB_GROUP_SIZE(SG_SIZE)
__attribute__((reqd_work_group_size(SG_SIZE, 1, 1)))
KERNEL(arg_max_min_opt)(const __global UNIT_TYPE* input, __global uint* output)
{
    const uint input_size = INPUT0_FEATURE_NUM * INPUT0_SIZE_X * INPUT0_SIZE_Y;
    const uint gid = get_group_id(0);
    const uint lid = get_sub_group_local_id();

    UNIT_TYPE input_blocks[INB_ARRAY_SIZE];
    uint indices[INB_ARRAY_SIZE];

    // Read INB_ARRAY_SIZE * SG_SIZE elements (cache them in registers, fill unaligned/unpadded data with
    //                                         UNIT_FILL_VAL).
    //
    // gid * INB_ARRAY_SIZE * SG_SIZE + (INB_ARRAY_SIZE - 1) * SG_SIZE + SG_SIZE - 1 < input_size
    // gid * INB_ARRAY_SIZE * SG_SIZE + INB_ARRAY_SIZE * SG_SIZE <= input_size
    // (gid + 1) * INB_ARRAY_SIZE * SG_SIZE <= input_size
    // (gid + 1) <= input_size / (INB_ARRAY_SIZE * SG_SIZE)   ->   as gid is integral, the floor is not an issue
    if (gid + 1 <= input_size / (INB_ARRAY_SIZE * SG_SIZE))
    {
        unroll_for(uint ai = 0; ai < INB_ARRAY_SIZE; ++ai)
        {
            // Can be exchanged with sub-group block read to INB_ARRAY_SIZE-component vector.
            input_blocks[ai] = input[gid * INB_ARRAY_SIZE * SG_SIZE + ai * SG_SIZE + lid];
            indices[ai] = gid * INB_ARRAY_SIZE * SG_SIZE + ai * SG_SIZE + lid;
        }
    }
    else
    {
        const uint last_gid = input_size / (INB_ARRAY_SIZE * SG_SIZE);

        uint ai = 0;
        unroll_for(uint last_base_off = last_gid * INB_ARRAY_SIZE * SG_SIZE; last_base_off + SG_SIZE <= input_size; last_base_off += SG_SIZE)
        {
            // Can be exchanged with sub-group block read to scalar.
            input_blocks[ai] = input[last_base_off + lid];
            indices[ai++] = last_base_off + lid;
        }

        const uint remainder_off = input_size / SG_SIZE * SG_SIZE;

        if (remainder_off < input_size)
        {
            input_blocks[ai] = lid < input_size - remainder_off ? input[remainder_off + lid] : UNIT_FILL_VAL;
            indices[ai++] = lid < input_size - remainder_off ? remainder_off + lid : 0;
        }

        unroll_for(; ai < INB_ARRAY_SIZE; ++ai)
        {
            input_blocks[ai] = UNIT_FILL_VAL;
        }
    }


    // Sort TOP_K elements (by linear scan and insert).
    const uint minmax_acc_array_size = (TOP_K + SG_SIZE - 1) / SG_SIZE;
    UNIT_TYPE acc[minmax_acc_array_size];
    uint result[minmax_acc_array_size];

    unroll_for (uint ai = 0; ai < minmax_acc_array_size; ++ai)
    {
        acc[ai] = UNIT_FILL_VAL;
        result[ai] = 0;
    }

    //__attribute__((opencl_unroll_hint))
    __attribute__((opencl_unroll_hint(1)))
    for (uint ii = 0; ii < INB_ARRAY_SIZE * SG_SIZE; ++ii)
    {
        UNIT_TYPE in_val = _sub_group_shuffle(input_blocks[ii / SG_SIZE], ii % SG_SIZE);
        uint in_index = _sub_group_shuffle(input_blocks[ii / SG_SIZE], ii % SG_SIZE);
        unroll_for(uint ai = 0; ai < minmax_acc_array_size; ++ai)
        {
            bool insert_flag = (in_val OP_ARG_REL acc[ai]);
            if (sub_group_any(insert_flag))
            {
                unroll_for(uint aj = minmax_acc_array_size; aj > ai + 1; --aj)
                {
                    acc[aj - 1] = _sub_group_shuffle_up(acc[aj - 2], acc[aj - 1], 1);
                    result[aj - 1] = _sub_group_shuffle_up(result[aj - 2], acc[aj - 1], 1);
                }
                UNIT_TYPE in_val_acc_mask = select(in_val, acc[ai], insert_flag);
                uint in_index_mask = select(in_index, result[ai], insert_flag);
                acc[ai] = select(acc[ai], _sub_group_shuffle_up(in_val, in_val_acc_mask, 1), insert_flag);
                result[ai] = select(result[ai], _sub_group_shuffle_up(in_index, in_index_mask, 1), insert_flag);
                break;
            }
        }
    }


    // Write TOP_K sorted results.
    uint ai = 0;
    unroll_for (uint k_base_off = 0; k_base_off + SG_SIZE <= TOP_K; k_base_off += SG_SIZE)
    {
        output[k_base_off + lid] = result[ai++] % input_size;
    }

    const uint k_remainder_off = TOP_K / SG_SIZE * SG_SIZE;
    if (k_remainder_off < TOP_K && lid < TOP_K - k_remainder_off)
    {
        output[k_remainder_off + lid] = result[ai] % input_size;
    }
}


#ifdef SG_SIZE_NEEDSUNDEF_
    #undef SG_SIZE
    #undef SG_SIZE_NEEDSUNDEF_
#endif
#ifdef INB_ARRAY_SIZE_NEEDSUNDEF_
    #undef INB_ARRAY_SIZE
    #undef INB_ARRAY_SIZE_NEEDSUNDEF_
#endif
#ifdef UNIT_FILL_VAL_NEEDSUNDEF_
    #undef UNIT_FILL_VAL
    #undef UNIT_FILL_VAL_NEEDSUNDEF_
#endif
#undef OP_ARG_REL
