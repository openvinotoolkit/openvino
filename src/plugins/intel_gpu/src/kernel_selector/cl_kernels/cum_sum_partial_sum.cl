// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/fetch_utils.cl"

inline void FUNC(get_indices)(int *axes)
{
    // 0 - batch
    // 1 - features
    // 2 - w
    // 3 - z
    // 4 - y
    // 5 - x
    axes[AXIS] = (uint)get_global_id(0);
#if AXIS == 0
    axes[1] = (uint)get_global_id(1) / OUTPUT_SIZE_W;
    axes[2] = (uint)get_global_id(1) % OUTPUT_SIZE_W;
    axes[3] = (uint)get_global_id(2) / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    const uint yx = (uint)get_global_id(2) % (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    axes[4] = yx / OUTPUT_SIZE_X;
    axes[5] = yx % OUTPUT_SIZE_X;
#elif AXIS == 1
    axes[0] = (uint)get_global_id(1) / OUTPUT_SIZE_W;
    axes[2] = (uint)get_global_id(1) % OUTPUT_SIZE_W;
    axes[3] = (uint)get_global_id(2) / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    const uint yx = (uint)get_global_id(2) % (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    axes[4] = yx / OUTPUT_SIZE_X;
    axes[5] = yx % OUTPUT_SIZE_X;
#elif AXIS == 2
    axes[0] = (uint)get_global_id(1) / OUTPUT_FEATURE_NUM;
    axes[1] = (uint)get_global_id(1) % OUTPUT_FEATURE_NUM;
    axes[3] = (uint)get_global_id(2) / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    const uint yx = (uint)get_global_id(2) % (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    axes[4] = yx / OUTPUT_SIZE_X;
    axes[5] = yx % OUTPUT_SIZE_X;
#elif AXIS == 3
    axes[0] = (uint)get_global_id(1) / OUTPUT_FEATURE_NUM;
    axes[1] = (uint)get_global_id(1) % OUTPUT_FEATURE_NUM;
    axes[2] = (uint)get_global_id(2) / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    const uint yx = (uint)get_global_id(2) % (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    axes[4] = yx / OUTPUT_SIZE_X;
    axes[5] = yx % OUTPUT_SIZE_X;
#elif AXIS == 4
    axes[0] = (uint)get_global_id(1) / OUTPUT_FEATURE_NUM;
    axes[1] = (uint)get_global_id(1) % OUTPUT_FEATURE_NUM;
    axes[2] = (uint)get_global_id(2) / (OUTPUT_SIZE_X * OUTPUT_SIZE_Z);
    const uint zx = (uint)get_global_id(2) % (OUTPUT_SIZE_X * OUTPUT_SIZE_Z);
    axes[3] = zx / OUTPUT_SIZE_X;
    axes[5] = zx % OUTPUT_SIZE_X;
#else
    axes[0] = (uint)get_global_id(1) / OUTPUT_FEATURE_NUM;
    axes[1] = (uint)get_global_id(1) % OUTPUT_FEATURE_NUM;
    axes[2] = (uint)get_global_id(2) / (OUTPUT_SIZE_Y * OUTPUT_SIZE_Z);
    const uint zy = (uint)get_global_id(2) % (OUTPUT_SIZE_Y * OUTPUT_SIZE_Z);
    axes[3] = zy / OUTPUT_SIZE_Y;
    axes[4] = zy % OUTPUT_SIZE_Y;
#endif
}

#if CUM_SUM_PARTIAL_SUM
inline uint FUNC(get_current_index)(int axis, int i)
{
#ifdef REVERSE
    return (uint)get_global_size(0)*(uint)get_local_size(0) - axis * (uint)get_local_size(0) - i - 1;
#else
    return axis * (uint)get_local_size(0) + i;
#endif
}

REQD_SUB_GROUP_SIZE(SIMD)
__attribute__((reqd_work_group_size(LWS, 1, 1)))
KERNEL(cum_sum_partial_sum)(
    const __global INPUT0_TYPE* input,
    __global PARTIAL_TYPE* partial)
{
    int axes[6], initial_axes[6];
    FUNC_CALL(get_indices)(axes);
    initial_axes[0] = axes[0];
    initial_axes[1] = axes[1];
    initial_axes[2] = axes[2];
    initial_axes[3] = axes[3];
    initial_axes[4] = axes[4];
    initial_axes[5] = axes[5];

    int exclusive = 0;
#ifdef EXCLUSIVE
#ifdef REVERSE
    ++exclusive;
#else
    --exclusive;
#endif
#endif

    INPUT0_TYPE res[BLOCK_SIZE];
    INPUT0_TYPE prev = 0;
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        axes[AXIS] = FUNC_CALL(get_current_index)(initial_axes[AXIS], i) + exclusive;
        uint ind = FUNC_CALL(get_input_index)(axes[0], axes[1], axes[2], axes[3], axes[4], axes[5]);
        res[i] = (axes[AXIS] < SUM_ITEMS_NUM && axes[AXIS] >= 0) ? input[ind] : INPUT0_VAL_ZERO;
        res[i] += prev;
        prev = res[i];
    }

    for (int i = 0; i < BLOCK_SIZE; ++i) {
        axes[AXIS] = FUNC_CALL(get_current_index)(initial_axes[AXIS], i);
        uint out_ind = FUNC_CALL(get_input_index)(axes[0], axes[1], axes[2], axes[3], axes[4], axes[5]);
        if (axes[AXIS] < SUM_ITEMS_NUM)
            partial[out_ind] = TO_PARTIAL_TYPE(res[i]);
    }
}
#else
inline uint FUNC(get_block_num)(int axis)
{
#ifdef REVERSE
    return (SUM_ITEMS_NUM - axis - 1) / BLOCK_SIZE;
#else
    return axis / BLOCK_SIZE;
#endif
}

// This function works incorrect for the last block when there are leftovers (i.e. SUM_ITEMS_NUM % BLOCKSIZE != 0)
// and REVERSE == false. But it is expected, since it will never be called for the last block when calculating 
// sum of the previous blocks (see loop in cum_sum_final), thus, no need to make it correct 
// at cost of complexity and performance.
inline uint FUNC(get_last_index_in_block)(int block)
{
    const int num_items_in_blocks_before = (block + 1) * BLOCK_SIZE;
#ifdef REVERSE
    return SUM_ITEMS_NUM - num_items_in_blocks_before;
#else
    return num_items_in_blocks_before - 1;
#endif
}

// main
REQD_SUB_GROUP_SIZE(SIMD)
__attribute__((reqd_work_group_size(LWS, 1, 1)))
KERNEL(cum_sum_final)(
    const __global PARTIAL_TYPE* partial,
    __global OUTPUT_TYPE* output)
{
    int axes[6];
    FUNC_CALL(get_indices)(axes);
    const uint batch = axes[0];
    const uint features = axes[1];
    const uint w = axes[2];
    const uint z = axes[3];
    const uint y = axes[4];
    const uint x = axes[5];

    uint ind = FUNC_CALL(get_input_index)(axes[0], axes[1], axes[2], axes[3], axes[4], axes[5]);
    PARTIAL_TYPE res = partial[ind];

    PARTIAL_TYPE sum = 0;
    const uint current_block = FUNC_CALL(get_block_num)(axes[AXIS]);

    for (int block = 0; block < current_block; ++block) {
        axes[AXIS] = FUNC_CALL(get_last_index_in_block)(block);
        ind = FUNC_CALL(get_input_index)(axes[0], axes[1], axes[2], axes[3], axes[4], axes[5]);
        sum += partial[ind];
    }

    const uint out_ind = FUNC_CALL(get_output_index)(batch, features, w, z, y, x);
    output[out_ind] = ACTIVATION(TO_OUTPUT_TYPE(res + sum), ACTIVATION_PARAMS);
}
#endif
