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

KERNEL(convolution_grad_weights_gpu_3x3)(
    const __global UNIT_TYPE* input_grad,
    __global UNIT_TYPE* output,
    __global UNIT_TYPE* filter,
#if BIAS_TERM
    __global UNIT_TYPE* bias,
#endif
#if MOMENTUM
    __global UNIT_TYPE* prev_grad_w,
#if BIAS_TERM
    __global UNIT_TYPE* prev_grad_b,
#endif
#endif
    const __global UNIT_TYPE* input,
    uint split_idx,
    float lr)
{
    const uint ofm = get_global_id(0);
    const uint ifm = get_global_id(1);

    if (ofm >= INPUT0_FEATURE_NUM || ifm >= INPUT1_FEATURE_NUM)
        return;

    const int in_x = -PADDING_SIZE_X;
    const int in_y = -PADDING_SIZE_Y;

    ACCUMULATOR_TYPE grad_w[9] = {};
#if BIAS_TERM
    ACCUMULATOR_TYPE grad_b = 0;
#endif

    uint weights_idx = ofm * FILTER_OFM_PITCH + ifm * FILTER_IFM_PITCH;

    for(int b = 0; b < INPUT0_BATCH_NUM; b++)
    {
        const uint grad_split_offset = split_idx * INPUT0_FEATURE_PITCH * FILTER_OFM_NUM;
        const uint in_split_offset = split_idx * INPUT1_FEATURE_PITCH * FILTER_IFM_NUM;

        for (uint i = 0; i < INPUT0_SIZE_Y; i++)
        {

            for (uint j = 0; j < INPUT0_SIZE_X; j+=2)
            {
                float2 grad;
                if (j + 1 >= INPUT0_SIZE_X)
                {
                    uint input_grad_idx = grad_split_offset + b*INPUT0_BATCH_PITCH + ofm*INPUT0_FEATURE_PITCH + j*INPUT0_X_PITCH + i*INPUT0_Y_PITCH;
                    grad.s0 = input_grad[input_grad_idx];
                    grad.s1 = 0;
                }
                else
                {
                    uint input_grad_idx = grad_split_offset + b*INPUT0_BATCH_PITCH + ofm*INPUT0_FEATURE_PITCH + j*INPUT0_X_PITCH + i*INPUT0_Y_PITCH;
                    grad = vload2(0, &input_grad[input_grad_idx]);
                }
                for (uint y = 0; y < 3; y++)
                {
                    const int input_offset_y = in_y + y + i;
                    const bool zero_y = input_offset_y >= INPUT1_SIZE_Y || input_offset_y < 0;
                    const int input_offset_x = in_x + j;
                    const bool zero_x = input_offset_x < 0 || input_offset_x + 3 >= INPUT1_SIZE_X;
                    uint input_idx = in_split_offset + b*INPUT1_BATCH_PITCH + ifm*INPUT1_FEATURE_PITCH + (uint)input_offset_x*INPUT1_X_PITCH + (uint)input_offset_y*INPUT1_Y_PITCH;
                    union v4 {
                        float s[4];
                        float4 v;
                    };
                    union v4 inp;
                    if (zero_y)
                        continue;
                    if (zero_x)
                    {
                        for (uint k = 0; k < 4; k++)
                        {
                            if (input_offset_x + k >= INPUT1_SIZE_X || input_offset_x + k < 0)
                                inp.s[k] = 0;
                            else
                                inp.s[k] = input[input_idx + k];
                        }
                    }
                    else
                    {
                        inp.v = vload4(0, &input[input_idx]);
                    }
                    for (uint x = 0; x < 3; x++)
                    {
                        grad_w[y * 3 + x] = mad(inp.s[x] * lr, grad.s0, grad_w[y * 3 + x]);
                        grad_w[y * 3 + x] = mad(inp.s[x + 1] * lr, grad.s1, grad_w[y * 3 + x]);
                    }
                }
#if BIAS_TERM
                grad_b += grad.s0;
                grad_b += grad.s1;
#endif
            }
        }
    }

    union {
        float  s[8];
        float8 v;
    } uweights_0_7;
    float uweights8;

#if MOMENTUM
    float dwa[9];
    uweights_0_7.v = vload8(0, &prev_grad_w[weights_idx]);
    dwa[0 * 3 + 0] = uweights_0_7.v.s0;
    dwa[0 * 3 + 1] = uweights_0_7.v.s1;
    dwa[0 * 3 + 2] = uweights_0_7.v.s2;
    dwa[1 * 3 + 0] = uweights_0_7.v.s3;
    dwa[1 * 3 + 1] = uweights_0_7.v.s4;
    dwa[1 * 3 + 2] = uweights_0_7.v.s5;
    dwa[2 * 3 + 0] = uweights_0_7.v.s6;
    dwa[2 * 3 + 1] = uweights_0_7.v.s7;
    dwa[2 * 3 + 2] = prev_grad_w[weights_idx + 8];
#endif

    uweights_0_7.v = vload8(0, &filter[weights_idx]);
    uweights8 = filter[weights_idx + 8];

#if MOMENTUM
    float8 newDelta_0_7 = (float8)(    
                                    grad_w[0 * 3 + 0] + (MOMENTUM_FACTOR * dwa[0 * 3 + 0]), 
                                    grad_w[0 * 3 + 1] + (MOMENTUM_FACTOR * dwa[0 * 3 + 1]),
                                    grad_w[0 * 3 + 2] + (MOMENTUM_FACTOR * dwa[0 * 3 + 2]), 
                                    grad_w[1 * 3 + 0] + (MOMENTUM_FACTOR * dwa[1 * 3 + 0]), 
                                    grad_w[1 * 3 + 1] + (MOMENTUM_FACTOR * dwa[1 * 3 + 1]),
                                    grad_w[1 * 3 + 2] + (MOMENTUM_FACTOR * dwa[1 * 3 + 2]), 
                                    grad_w[2 * 3 + 0] + (MOMENTUM_FACTOR * dwa[2 * 3 + 0]),
                                    grad_w[2 * 3 + 1] + (MOMENTUM_FACTOR * dwa[2 * 3 + 1]));
    float newDelta8 =               grad_w[2 * 3 + 2] + (MOMENTUM_FACTOR * dwa[2 * 3 + 2]);
#else
    float8 newDelta_0_7 = (float8)(    
                                    grad_w[0 * 3 + 0], 
                                    grad_w[0 * 3 + 1],
                                    grad_w[0 * 3 + 2], 
                                    grad_w[1 * 3 + 0], 
                                    grad_w[1 * 3 + 1],
                                    grad_w[1 * 3 + 2], 
                                    grad_w[2 * 3 + 0],
                                    grad_w[2 * 3 + 1]);
    float newDelta8 =               grad_w[2 * 3 + 2];    
#endif
    uweights8      -= newDelta8;
    uweights_0_7.v -= newDelta_0_7;

    vstore8(uweights_0_7.v, 0, &filter[weights_idx]);
    filter[weights_idx + 8] = uweights8;
#if MOMENTUM
    vstore8(newDelta_0_7, 0, &prev_grad_w[weights_idx]);
    prev_grad_w[weights_idx + 8] = newDelta8;
#endif

#if BIAS_TERM
    if(ifm == 0)
    {
#if MOMENTUM
        UNIT_TYPE update_gradient_b = lr * grad_b + prev_grad_b[ofm] * MOMENTUM_FACTOR;
        bias[ofm] -= update_gradient_b;
        prev_grad_b[ofm] = update_gradient_b;
#else
        bias[ofm] -= lr * grad_b;
#endif
    }
#endif
}
