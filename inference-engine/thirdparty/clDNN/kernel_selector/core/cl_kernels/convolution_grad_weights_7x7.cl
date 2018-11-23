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

KERNEL(convolution_grad_weights_gpu_7x7)(
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
    const uint x_filter = get_global_id(0);
    const uint ofm = get_global_id(1);
    const uint ifm = get_global_id(2);

    if (x_filter >= 7 || ofm >= INPUT0_FEATURE_NUM || ifm >= INPUT1_FEATURE_NUM)
        return;

    const int in_x = -PADDING_SIZE_X;
    const int in_y = -PADDING_SIZE_Y;

    ACCUMULATOR_TYPE grad_w[7] = { 0, 0, 0, 0, 0, 0, 0 };
#if BIAS_TERM
    ACCUMULATOR_TYPE grad_b = UNIT_VAL_ZERO;
#endif

    uint weights_idx = ofm * FILTER_OFM_PITCH + ifm * FILTER_IFM_PITCH;

    for(int b = 0; b < INPUT0_BATCH_NUM; b++)
    {
        const uint grad_split_offset = split_idx * INPUT0_FEATURE_PITCH * FILTER_OFM_NUM;
        const uint in_split_offset = split_idx * INPUT1_FEATURE_PITCH * FILTER_IFM_NUM;

        for(int i = 0; i < INPUT0_SIZE_Y; i++)
        {
            for(int j = 0; j < INPUT0_SIZE_X; j++)
            {
                float grad;
                uint input_grad_idx = grad_split_offset + b*INPUT0_BATCH_PITCH + ofm*INPUT0_FEATURE_PITCH + j*INPUT0_X_PITCH + i*INPUT0_Y_PITCH;
                grad = input_grad[input_grad_idx];
                for(uint y_filter = 0; y_filter < 7; y_filter++)
                {
                    const int input_offset_y = in_y + y_filter + i * STRIDE_SIZE_Y;
                    const bool zero_y = input_offset_y >= INPUT1_SIZE_Y || input_offset_y < 0;
                    const int input_offset_x = in_x + x_filter + j * STRIDE_SIZE_X;
                    const bool zero_x = input_offset_x < 0 || input_offset_x >= INPUT1_SIZE_X;
                    uint input_idx = in_split_offset + b*INPUT1_BATCH_PITCH + ifm*INPUT1_FEATURE_PITCH + (uint)input_offset_x*INPUT1_X_PITCH + (uint)input_offset_y*INPUT1_Y_PITCH;
                    if(!zero_x && !zero_y)
                    {
                        const float delta_f = input[input_idx] * lr * grad;
                        grad_w[y_filter] += delta_f;
                    }
                } 
#if BIAS_TERM
                grad_b += grad;
#endif
            }
        }
    }
    for(uint y_filter = 0; y_filter < 7; y_filter++)
    {
        uint address = weights_idx + 48 - (7 * (6 - y_filter) + (6 - x_filter));
#if MOMENTUM
        float dw = prev_grad_w[address];
        const float delta_f_m = MOMENTUM_FACTOR * dw;
        grad_w[y_filter] += delta_f_m;
        prev_grad_w[address] = grad_w[y_filter];
#endif
        filter[address] -= grad_w[y_filter];
    }
#if BIAS_TERM
    if(ifm == 0 && x_filter == 0)
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