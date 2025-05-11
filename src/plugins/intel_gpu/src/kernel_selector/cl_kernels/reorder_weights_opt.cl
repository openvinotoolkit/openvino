// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

INIT_INPUT0_INDEX_FUNC_HERE
INIT_OUTPUT_INDEX_FUNC_HERE

#if OUTPUT_GROUPED
#   if OUTPUT_DIMS == 5
#       define IDX_ORDER g, o, i, y, x
#       define BLOCK_IDX_ORDER g, o_blocked, i_blocked, y, x
#   elif OUTPUT_DIMS == 6
#       define IDX_ORDER g, o, i, z, y, x
#       define BLOCK_IDX_ORDER g, o_blocked, i_blocked, z, y, x
#   endif
#else
#   if OUTPUT_DIMS == 4
#       define IDX_ORDER o, i, y, x
#       define BLOCK_IDX_ORDER o_blocked, i_blocked, y, x
#   elif OUTPUT_DIMS == 5
#       define IDX_ORDER o, i, z, y, x
#       define BLOCK_IDX_ORDER o_blocked, i_blocked, z, y, x
#   endif
#endif
#define GET_INDEX(PREFIX, ORDER) CAT(PREFIX, _GET_INDEX)(ORDER)

#if OSV_FIRST
#   define FIRST_BLOCK_SIZE OFM_BLOCK_SIZE
#   define SECOND_BLOCK_SIZE IFM_BLOCK_SIZE
#   define PITCH INPUT0_IFM_PITCH
#   define SECOND_SIZE IFM_SIZE
#else
#   define FIRST_BLOCK_SIZE IFM_BLOCK_SIZE
#   define SECOND_BLOCK_SIZE OFM_BLOCK_SIZE
#   define PITCH INPUT0_OFM_PITCH
#   define SECOND_SIZE OFM_SIZE
#endif

#define OUTPUT_VEC_TYPE MAKE_VECTOR_TYPE(OUTPUT_TYPE, SECOND_BLOCK_SIZE)
#define OUTPUT_BLOCK_WRITE(ptr, offset, val) BLOCK_WRITEN(OUTPUT_TYPE, SECOND_BLOCK_SIZE, ptr, offset, val)

REQD_SUB_GROUP_SIZE(FIRST_BLOCK_SIZE)
__attribute__((reqd_work_group_size(1, 1, FIRST_BLOCK_SIZE)))
KERNEL(reorder_weights_opt)(const __global INPUT0_TYPE* input, __global OUTPUT_TYPE* output)
{
    const int lid = get_sub_group_local_id();
    const int g_io = get_global_id(0);
#if OSV_FIRST
#if OUTPUT_GROUPED
    const int i = (g_io % (OUTPUT_IFM_NUM / SECOND_BLOCK_SIZE)) * SECOND_BLOCK_SIZE;
    const int g = (g_io / (OUTPUT_IFM_NUM / SECOND_BLOCK_SIZE));
#else
    const int i = g_io * SECOND_BLOCK_SIZE;
#endif  // OUTPUT_GROUPED
    const int o_blocked = (int)get_group_id(2) * FIRST_BLOCK_SIZE;
    const int o = o_blocked + lid;
    const int i_blocked = i;
#else  // OSV_FIRST
#if OUTPUT_GROUPED
    const int o = (g_io % (OUTPUT_OFM_NUM / SECOND_BLOCK_SIZE)) * SECOND_BLOCK_SIZE;
    const int g = (g_io / (OUTPUT_OFM_NUM / SECOND_BLOCK_SIZE));
#else
    const int o = g_io * SECOND_BLOCK_SIZE;
#endif  // OUTPUT_GROUPED
    const int i_blocked = (int)get_group_id(2) * FIRST_BLOCK_SIZE;
    const int i = i_blocked + lid;
    const int o_blocked = o;
#endif  // OSV_FIRST

    const int zyx = get_global_id(1);
    const int x = zyx % OUTPUT_SIZE_X;
#if (OUTPUT_DIMS - OUTPUT_GROUPED) == 5
    const int y = zyx / OUTPUT_SIZE_X % OUTPUT_SIZE_Y;
    const int z = zyx / OUTPUT_SIZE_X / OUTPUT_SIZE_Y;
#else
    const int y = zyx / OUTPUT_SIZE_X;
#endif  // (OUTPUT_DIMS - OUTPUT_GROUPED) == 5

    int input_idx = GET_INDEX(INPUT0, IDX_ORDER);
    const int output_idx = GET_INDEX(OUTPUT, BLOCK_IDX_ORDER);

#if SECOND_BLOCK_SIZE == 1
    const OUTPUT_TYPE val = TO_OUTPUT_TYPE(input[input_idx]);
#else
    OUTPUT_VEC_TYPE val = 0;
    unroll_for (int b = 0; b < SECOND_BLOCK_SIZE; b++) {
        val[b] = TO_OUTPUT_TYPE(input[input_idx]);
        input_idx += PITCH;
    }
#endif  // SECOND_BLOCK_SIZE == 1
#if OUTPUT_LEFTOVERS
#if OSV_FIRST
    const bool doWrite = o < OUTPUT_OFM_NUM;
    if (o_blocked >= OUTPUT_OFM_NUM - FIRST_BLOCK_SIZE) {
#else
    const bool doWrite = i < OUTPUT_IFM_NUM;
    if (i_blocked >= OUTPUT_IFM_NUM - FIRST_BLOCK_SIZE) {
#endif  // OSV_FIRST
#if SECOND_BLOCK_SIZE > 1
        unroll_for(int b = 0; b < SECOND_BLOCK_SIZE; b++)
            if (doWrite)
                output[output_idx + b * SECOND_SIZE + lid] = val[b];
#else
            if (doWrite)
                output[output_idx + lid] = val;
#endif  // SECOND_BLOCK_SIZE > 1
    }
    else
#endif  // OUTPUT_LEFTOVERS
    {
        OUTPUT_BLOCK_WRITE(output, output_idx, val);
    }
}

#undef OUTPUT_VEC_TYPE
#undef OSV_FIRST
#undef FIRST_BLOCK_SIZE
#undef SECOND_BLOCK_SIZE
#undef PITCH
#undef SECOND_SIZE
#undef GET_INDEX
#undef BLOCK_IDX_ORDER
#undef IDX_ORDER
