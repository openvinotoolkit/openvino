// Copyright (c) 2018 Intel Corporation
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

#define LOCAL_SIZE INPUT0_BATCH_NUM

__attribute__((reqd_work_group_size(LOCAL_SIZE, 1, 1)))
KERNEL(batch_norm_grad_gpu)(const __global UNIT_TYPE* input_grad, __global UNIT_TYPE* input, __global UNIT_TYPE* inv_var,  __global UNIT_TYPE* output)
{
    __local ACCUMULATOR_TYPE grad_sum[LOCAL_SIZE];
    __local ACCUMULATOR_TYPE grad_sum_in[LOCAL_SIZE];

    const uint local_idx = (uint)get_local_id(0);
    const uint f = (uint)get_global_id(1);

    grad_sum[local_idx] = 0;
    grad_sum_in[local_idx] = 0;

    uint grad_idx = GET_DATA_INDEX(INPUT0, local_idx, f, 0, 0);
    for (uint y = 0; y < INPUT0_SIZE_Y; y++)
    {
        for (uint x = 0; x < INPUT0_SIZE_X; x++)
        {
            UNIT_TYPE in_g = input_grad[grad_idx];
            grad_sum[local_idx] += in_g;
            grad_sum_in[local_idx] += in_g * input[grad_idx]; 
            grad_idx += INPUT0_X_PITCH;
        }
        grad_idx += INPUT0_Y_PITCH - INPUT0_SIZE_X * INPUT0_X_PITCH;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for(uint offset = LOCAL_SIZE / 2; offset > 0; offset /= 2) 
    {
        if (local_idx < offset) 
        {
            grad_sum[local_idx] += grad_sum[local_idx + offset];
            grad_sum_in[local_idx] += grad_sum_in[local_idx + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    UNIT_TYPE grad_mean = grad_sum[0] / (OUTPUT_BATCH_NUM * OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    UNIT_TYPE grad_mean_in = grad_sum_in[0] / (OUTPUT_BATCH_NUM * OUTPUT_SIZE_X * OUTPUT_SIZE_Y);

    uint out_idx = GET_DATA_INDEX(OUTPUT, local_idx, f, 0, 0);
    for (uint y = 0; y < OUTPUT_SIZE_Y; y++)
    {
        for (uint x = 0; x < OUTPUT_SIZE_X; x++)
        {
            UNIT_TYPE grad_out = inv_var[f] * (input_grad[out_idx] - grad_mean - grad_mean_in * input[out_idx]);

            if (grad_out > 5.0f)
                grad_out = 5.0f;
            else if (grad_out < -5.0f)
                grad_out = -5.0f;
            
            output[out_idx] = grad_out;
            out_idx += OUTPUT_X_PITCH;
        }
        out_idx += OUTPUT_Y_PITCH - OUTPUT_SIZE_X * OUTPUT_X_PITCH;
    }

}

#undef LOCAL_SIZE