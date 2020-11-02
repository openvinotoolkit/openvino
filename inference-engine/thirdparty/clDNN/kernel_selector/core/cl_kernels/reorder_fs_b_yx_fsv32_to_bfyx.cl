// Copyright (c) 2019 Intel Corporation
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

#include "include/fetch.cl"
#include "include/unit_type.cl"

__attribute__((reqd_work_group_size(1, LWS1, 1)))
KERNEL (reorder_fs_b_yx_fsv32_to_bfyx)(
        const __global INPUT_REORDER_TYPE* input,
        __global OUTPUT_REORDER_TYPE* output
)
{
    const uint b = get_global_id(0);
    const uint f_blocked = get_group_id(1) * LWS1;
    const uint sglid = get_local_id(1);
    const uint f = f_blocked + sglid;
    const uint y = ((uint)get_global_id(2) / X_BLOCKED_SIZE);
    const uint x = ((uint)get_global_id(2) % X_BLOCKED_SIZE) * X_BLOCK_SIZE;
    __local CALC_TYPE in_data[X_BLOCK_SIZE * LWS1];

    const uint in_idx = INPUT0_GET_INDEX(b, f_blocked, y, x);

    __attribute__((opencl_unroll_hint(X_BLOCK_SIZE)))
    for (int i = 0; i < X_BLOCK_SIZE; i++) {
        in_data[sglid * X_BLOCK_SIZE + i] = input[in_idx + (i * FSV) + sglid];
    }

    __attribute__((opencl_unroll_hint(X_BLOCK_SIZE)))
    for (int i = 0; i < X_BLOCK_SIZE; i++) {
        const uint f_idx = f_blocked + i * (LWS1 / X_BLOCK_SIZE) + (sglid / X_BLOCK_SIZE);
        const uint x_idx = x + (sglid % X_BLOCK_SIZE);
        const uint out_idx = OUTPUT_GET_INDEX(b, f_idx, y, x_idx);
        const uint data_idx = sglid % X_BLOCK_SIZE + (sglid / X_BLOCK_SIZE) * X_BLOCK_SIZE + i * (LWS1 / X_BLOCK_SIZE) * X_BLOCK_SIZE;
#if defined(LEFTOVERS_OC) || defined(LEFTOVERS_OX)
#if defined(LEFTOVERS_OC) && defined(LEFTOVERS_OX)
        const bool skip = f_idx >= OUTPUT_FEATURE_NUM || x_idx >= OUTPUT_SIZE_X;
#elif defined(LEFTOVERS_OC)
        const bool skip = f_idx >= OUTPUT_FEATURE_NUM;
#else
        const bool skip = x_idx >= OUTPUT_SIZE_X;
#endif
        if (!skip) {
            output[out_idx] = ACTIVATION_TYPED(OUTPUT_REORDER, in_data[data_idx], ACTIVATION_PARAMS_TYPED);
        }
#else
        output[out_idx] = ACTIVATION_TYPED(OUTPUT_REORDER, in_data[data_idx], ACTIVATION_PARAMS_TYPED);
#endif
    }
}
