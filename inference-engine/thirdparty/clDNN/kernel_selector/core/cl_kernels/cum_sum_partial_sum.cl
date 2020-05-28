// Copyright (c) 2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "include/include_all.cl"

///////////////////////// Input Index /////////////////////////
inline uint FUNC(get_input_index)(uint b, uint f, uint w, uint z, uint y, uint x)
{
#if INPUT0_DIMS < 5
    return INPUT0_GET_INDEX(b, f, y, x);
#elif INPUT0_DIMS == 5
    return INPUT0_GET_INDEX(b, f, z, y, x);
#elif INPUT0_DIMS == 6
    return INPUT0_GET_INDEX(b, f, w, z, y, x);
#else
#error cum_sum_ref.cl: input format - not supported
#endif
}

///////////////////////// Output Index /////////////////////////
inline uint FUNC(get_output_index)(uint b, uint f, uint w, uint z, uint y, uint x)
{
#if OUTPUT_DIMS < 5
    return OUTPUT_GET_INDEX(b, f, y, x);
#elif OUTPUT_DIMS == 5
    return OUTPUT_GET_INDEX(b, f, z, y, x);
#elif OUTPUT_DIMS == 6
    return OUTPUT_GET_INDEX(b, f, w, z, y, x);
#else
#error cum_sum_ref.cl: output format - not supported
#endif
}

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

#define unroll_for __attribute__((opencl_unroll_hint)) for

#if CUM_SUM_PARTIAL_SUM
inline uint FUNC(get_current_index)(int axis, int i)
{
#ifdef REVERSE
    return (uint)get_global_size(0)*(uint)get_local_size(0) - axis * (uint)get_local_size(0) - i - 1;
#else
    return axis * (uint)get_local_size(0) + i;
#endif
}

__attribute__((intel_reqd_sub_group_size(SIMD)))
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

inline uint FUNC(get_current_index)(int i)
{
#ifdef REVERSE
    return SUM_ITEMS_NUM - i*BLOCK_SIZE - BLOCK_SIZE;
#else
    return i*BLOCK_SIZE + BLOCK_SIZE - 1;
#endif
}

// main
__attribute__((intel_reqd_sub_group_size(SIMD)))
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
    uint block_num = FUNC_CALL(get_block_num)(axes[AXIS]);
    int n = 4;
    for (int i = 0; i < block_num / n; ++i) {
        unroll_for (int j = 0; j < n; ++j) {
            axes[AXIS] = FUNC_CALL(get_current_index)(i*n + j);
            ind = FUNC_CALL(get_input_index)(axes[0], axes[1], axes[2], axes[3], axes[4], axes[5]);
            sum += partial[ind];
        }
    }

    uint out_ind = FUNC_CALL(get_output_index)(batch, features, w, z, y, x);
    output[out_ind] = ACTIVATION(TO_OUTPUT_TYPE(res + sum), ACTIVATION_PARAMS);
}
#endif
