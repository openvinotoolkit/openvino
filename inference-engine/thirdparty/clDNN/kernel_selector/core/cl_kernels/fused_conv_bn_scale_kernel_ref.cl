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
KERNEL(convolution)(
    __global INPUT0_TYPE* input, 
    __global OUTPUT_TYPE* output, 
    __global FILTER_TYPE* weights, 
#if BIAS_TERM
    __global BIAS_TYPE* biases,
#endif
    uint split_idx,
    __global INPUT0_TYPE* scale_in
#if SCALE_BIAS_TERM
    , __global INPUT0_TYPE* scale_bias
#endif
#if FUSED_TRAINING
    , __global INPUT0_TYPE* inv_var,
    __global INPUT0_TYPE* conv_output,
    __global INPUT0_TYPE* bn_output
#endif
    )
{
    const uint f = get_global_id(1);
    const uint b = get_global_id(0);

    UNIT_TYPE conv_out = UNIT_VAL_ZERO;

    const uint in_split_offset = split_idx * INPUT0_FEATURE_PITCH * FILTER_IFM_NUM;

    const uint filter_offset = f*FILTER_OFM_PITCH;
    const uint input_offset = b*INPUT0_BATCH_PITCH + INPUT0_OFFSET + in_split_offset;

    for (uint y = 0; y < OUTPUT_SIZE_Y; ++y)
    {
        const int input_y = y * STRIDE_SIZE_Y - PADDING_SIZE_Y;
        for (uint x = 0; x < OUTPUT_SIZE_X; ++x)
        {
            const int input_x = x * STRIDE_SIZE_X - PADDING_SIZE_X;
            for (uint k = 0; k < FILTER_IFM_NUM; ++k)
            {
                for (uint j = 0; j < FILTER_SIZE_Y ; ++j)
                {
                    const int input_offset_y = input_y + j * DILATION_SIZE_Y;
                    const bool zero_y = input_offset_y >= INPUT0_SIZE_Y || input_offset_y < 0;

                    if(!zero_y)
                    {
                        for (uint i = 0; i < FILTER_SIZE_X ; ++i)
                        {
                            const int input_offset_x = input_x + i * DILATION_SIZE_X;
                            const bool zero_x = input_offset_x >= INPUT0_SIZE_X || input_offset_x < 0;

                            if(!zero_x)
                            {
                                uint input_idx = input_offset + (uint)input_offset_x*INPUT0_X_PITCH + (uint)input_offset_y*INPUT0_Y_PITCH + k*INPUT0_FEATURE_PITCH;
                                uint filter_idx = filter_offset + k*FILTER_IFM_PITCH + j*FILTER_Y_PITCH + i*FILTER_X_PITCH;
                                conv_out += input[input_idx] * weights[filter_idx];       
                            }
                        }
                    }
                }
            }
#if BIAS_TERM
                conv_out += (UNIT_TYPE)biases[f];
#endif

                const uint out_split_offset = split_idx * OUTPUT_FEATURE_PITCH * OUTPUT_FEATURE_NUM;
                const uint dst_index = GET_DATA_INDEX(OUTPUT, b, f, y, x) + out_split_offset;
#ifdef FUSED_TRAINING
                conv_output[dst_index] = conv_out;
#else
                output[dst_index] = conv_out;
#endif
        }
    }


    // BATCH NORM PART
    barrier(CLK_LOCAL_MEM_FENCE);
    
    __local ACCUMULATOR_TYPE sum[LOCAL_SIZE];

    const uint local_idx = b;

    sum[local_idx] = 0;

    uint input_idx = GET_DATA_INDEX(OUTPUT, local_idx, f, 0, 0);
    for (uint y = 0; y < OUTPUT_SIZE_Y; y++)
    {
        for (uint x = 0; x < OUTPUT_SIZE_X; x++)
        {
#ifdef FUSED_TRAINING
            UNIT_TYPE in = conv_output[input_idx];
#else
            UNIT_TYPE in = output[input_idx];
#endif
            sum[local_idx] += in;
            input_idx += OUTPUT_X_PITCH;
        }
        input_idx += OUTPUT_Y_PITCH - OUTPUT_SIZE_X * OUTPUT_X_PITCH;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for(uint offset = LOCAL_SIZE / 2; offset > 0; offset /= 2) 
    {
        if (local_idx < offset) 
        {
            sum[local_idx] += sum[local_idx + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    UNIT_TYPE mean = sum[0] / (OUTPUT_BATCH_NUM * OUTPUT_SIZE_X * OUTPUT_SIZE_Y);

    sum[local_idx] = 0;

    input_idx = GET_DATA_INDEX(OUTPUT, local_idx, f, 0, 0);
    for (uint y = 0; y < OUTPUT_SIZE_Y; y++)
    {
        for (uint x = 0; x < OUTPUT_SIZE_X; x++)
        {
#ifdef FUSED_TRAINING
            UNIT_TYPE in = conv_output[input_idx] - mean;
#else
            UNIT_TYPE in = output[input_idx] - mean;
#endif
            sum[local_idx] += in * in;
            input_idx += OUTPUT_X_PITCH;
        }
        input_idx += OUTPUT_Y_PITCH - OUTPUT_SIZE_X * OUTPUT_X_PITCH;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for(uint offset = LOCAL_SIZE / 2; offset > 0; offset /= 2) 
    {
        if (local_idx < offset) 
        {
            sum[local_idx] += sum[local_idx + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float variance = sum[0] / (OUTPUT_BATCH_NUM * OUTPUT_SIZE_X * OUTPUT_SIZE_Y);

    float inv_variance = (float)(1.0 / sqrt(variance + EPSILON));

#ifdef FUSED_TRAINING
    if (local_idx == 0)
        inv_var[f] = inv_variance;
#endif

    uint out_idx = GET_DATA_INDEX(OUTPUT, local_idx, f, 0, 0);
    for (uint y = 0; y < OUTPUT_SIZE_Y; y++)
    {
        for (uint x = 0; x < OUTPUT_SIZE_X; x++)
        {
#ifdef FUSED_TRAINING
            UNIT_TYPE out_val = inv_variance * (conv_output[out_idx] - mean);
            bn_output[out_idx] = out_val;
#ifdef SCALE_BIAS_TERM
            output[out_idx] = ACTIVATION(out_val * scale_in[f] + scale_bias[f], NL_M, NL_N);  
#else
            output[out_idx] = ACTIVATION(out_val * scale_in[f], NL_M, NL_N);  
#endif
#else
#ifdef SCALE_BIAS_TERM
            output[out_idx] = ACTIVATION(inv_variance * (output[out_idx] - mean) * scale_in[f] + scale_bias[f], NL_M, NL_N);  
#else
            output[out_idx] = ACTIVATION(inv_variance * (output[out_idx] - mean) * scale_in[f], NL_M, NL_N);
#endif
#endif
            out_idx += OUTPUT_X_PITCH;
        }
        out_idx += OUTPUT_Y_PITCH - OUTPUT_SIZE_X * OUTPUT_X_PITCH;
    }

}

#undef LOCAL_SIZE