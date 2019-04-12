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
KERNEL(batch_norm_gpu)(
    const __global UNIT_TYPE* input,
	#ifdef MEAN_VAR_OUT
		__global UNIT_TYPE* mean_out,
		__global UNIT_TYPE* variance_out,
	#endif
	#ifdef SCALE_SHIFT
	     __global UNIT_TYPE* scale,
		 __global UNIT_TYPE* shift,
	#endif
	#ifdef FORWARD
		__global UNIT_TYPE* inv_var,
	#endif
       __global UNIT_TYPE* output)
{
    __local ACCUMULATOR_TYPE sum[LOCAL_SIZE];

    const uint local_idx = (uint)get_global_id(0);
    const uint f = (uint)get_global_id(1);

    sum[local_idx] = 0;

    uint input_idx = GET_DATA_INDEX(INPUT0, local_idx, f, 0, 0);
    for (uint y = 0; y < INPUT0_SIZE_Y; y++)
    {
        for (uint x = 0; x < INPUT0_SIZE_X; x++)
        {
            UNIT_TYPE in = input[input_idx];
            sum[local_idx] += in;
            input_idx += INPUT0_X_PITCH;
        }
        input_idx += INPUT0_Y_PITCH - INPUT0_SIZE_X * INPUT0_X_PITCH;
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
#ifdef MEAN_VAR_OUT
		mean_out[f] = mean;
#endif
    sum[local_idx] = 0;

    input_idx = GET_DATA_INDEX(INPUT0, local_idx, f, 0, 0);
    for (uint y = 0; y < INPUT0_SIZE_Y; y++)
    {
        for (uint x = 0; x < INPUT0_SIZE_X; x++)
        {
            UNIT_TYPE in = input[input_idx] - mean;
            sum[local_idx] += in * in;
            input_idx += INPUT0_X_PITCH;
        }
        input_idx += INPUT0_Y_PITCH - INPUT0_SIZE_X * INPUT0_X_PITCH;
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
#ifdef MEAN_VAR_OUT
	variance_out[f] = variance;
#endif
    float inv_variance = (float)(1.0 / sqrt(variance + EPSILON));
#ifdef FORWARD
    if (local_idx == 0)
        inv_var[f] = inv_variance;
#endif

    uint out_idx = GET_DATA_INDEX(OUTPUT, local_idx, f, 0, 0);
    for (uint y = 0; y < OUTPUT_SIZE_Y; y++)
    {
        for (uint x = 0; x < OUTPUT_SIZE_X; x++)
        {
			#ifdef SCALE_SHIFT
				output[out_idx] = (inv_variance * (input[out_idx] - mean)) * scale[f] + shift[f];
			#else
				output[out_idx] = inv_variance * (input[out_idx] - mean);
			#endif
            out_idx += OUTPUT_X_PITCH;
        }
        out_idx += OUTPUT_Y_PITCH - OUTPUT_SIZE_X * OUTPUT_X_PITCH;
    }
}

#undef LOCAL_SIZE