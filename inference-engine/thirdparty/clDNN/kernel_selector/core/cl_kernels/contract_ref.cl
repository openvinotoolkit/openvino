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

#include "include/include_all.cl"


KERNEL(contract_ref)(
    const __global INPUT0_TYPE* input,
    __global INPUT0_TYPE* output)
{
    INPUT0_TYPE out_val = REDUCE_SEED;

#if REDUCE_B
    for (uint in_b = 0; in_b < INPUT0_BATCH_NUM; ++in_b) {
#else
    const uint in_b = (uint) get_global_id(DIM_B);
#endif

#if REDUCE_F
    for (uint in_f = 0; in_f < INPUT0_FEATURE_NUM; ++in_f) {
#else
    const uint in_f = (uint) get_global_id(DIM_F);
#endif

#if REDUCE_Y
    for (uint in_y = 0; in_y < INPUT0_SIZE_Y; ++in_y) {
#else
    const uint in_y = (uint) get_global_id(DIM_Y);
#endif

#if REDUCE_X
    for (uint in_x = 0; in_x < INPUT0_SIZE_X; ++in_x) {
#else
    const uint in_x = (uint) get_global_id(DIM_X);
#endif

    out_val = REDUCE_OPERATION(out_val, input[GET_DATA_INDEX(INPUT0, in_b, in_f, in_y, in_x)]);

#if REDUCE_X
    }
#endif
#if REDUCE_Y
    }
#endif
#if REDUCE_F
    }
#endif
#if REDUCE_B
    }
#endif

    output[GET_DATA_INDEX(OUTPUT, 0, get_global_id(0), get_global_id(1), get_global_id(2))] = out_val;
}