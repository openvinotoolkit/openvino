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

KERNEL(scale_grad_weights_gpu_ref)(
    const __global UNIT_TYPE* input,
    const __global UNIT_TYPE* input_grad,
    __global OUTPUT_TYPE* output,
	__global UNIT_TYPE* scale,
#if BIAS_TERM
    __global UNIT_TYPE* bias,
#endif
#if MOMENTUM
    __global UNIT_TYPE* prev_grad_w,
#if BIAS_TERM
    __global UNIT_TYPE* prev_grad_b,
#endif
#endif
    const float lr
    )
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
            grad_sum[local_idx] += in_g * lr;
            grad_sum_in[local_idx] += in_g * input[grad_idx] * lr; 
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
    
    if (local_idx == 0)
    {
#if MOMENTUM
    UNIT_TYPE update_gradient_w = grad_sum_in[0] + prev_grad_w[f] * MOMENTUM_FACTOR + DECAY_RATE * lr * scale[f];
    scale[f] -= update_gradient_w;
    prev_grad_w[f] = update_gradient_w;
#else
    scale[f] -= grad_sum_in[0] + DECAY_RATE * lr * scale[f];
#endif

#if BIAS_TERM
#if MOMENTUM
    UNIT_TYPE update_gradient_b = prev_grad_b[f] * MOMENTUM_FACTOR + grad_sum[0];
    bias[f] -= update_gradient_b;
    prev_grad_b[f] = update_gradient_b;
#else
    bias[f] -= grad_sum[0];
#endif
#endif
    }  
}

#undef LOCAL_SIZE