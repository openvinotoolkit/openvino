// Copyright (c) 2016-2017 Intel Corporation
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

#ifndef DIRECTION
#define DIRECTION 0
#endif

#ifndef SIMD
#define SIMD 16
#endif

// Sums value of result across all subgroups.
#define SUM_ACROSS_SUB_GROUP(val) \
 \
{ \
    val += intel_sub_group_shuffle(val, x+1); \
    val += intel_sub_group_shuffle(val, x+2); \ 
    val += intel_sub_group_shuffle(val, x+4); \ 
    val += intel_sub_group_shuffle(val, x+8); \
} 

// input     = [    batch,  sequence,               1,      input_size ]
// weights   = [        1, direction, 4 * hidden_size,      input_size ]
// recurrent = [        1, direction, 4 * hidden_size,     hidden_size ]
// biases    = [        1,         1,       direction, 4 * hidden_size ] optional
// hidden    = [    batch, direction,               1,     hidden_size ] optional
// tempGEMM  = [    batch, direction,               1, 4 * hidden_size ] output

__attribute__((reqd_work_group_size(SIMD, 1, 1)))
KERNEL(lstm_gemm)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    const __global WEIGHTS_TYPE* weights
#if HIDDEN_TERM
    , const __global OUTPUT_TYPE* hidden,
    const __global RECURRENT_TYPE* recurrent
#endif
#if BIAS_TERM
    , const __global BIAS_TYPE* biases
#endif
    )
{
    const uint x = get_local_id(0);
    const uint y = get_global_id(1);
	const int local_sz = get_local_size(0);

	uint K; 	
	int start_offset;
	int end_offset;
	int matrix_offset;  
	int vector_offset; 
	float4 sum;
	float result;
	
	K = INPUT0_SIZE_X;  // Width of  weight matrix
	start_offset = GET_DATA_INDEX(WEIGHTS, 0, DIRECTION, y, 0);  // set as the starting offset of the weight matrix 
	end_offset = start_offset + K;
	matrix_offset = start_offset + (x * 4);  // Weight offset for the work item to work on
	vector_offset = GET_DATA_INDEX(INPUT0, 0, 0, INPUT_DIRECTION, (x*4));  // Input offset for the work item to work on
	sum = (float4)(0.f);
	result = 0;
	for(; matrix_offset < end_offset; matrix_offset += (local_sz * 4), vector_offset += (local_sz * 4))
	{
		half4 mask = (half4) (1 , (matrix_offset + 1) < end_offset , (matrix_offset + 2) < end_offset , (matrix_offset + 3) < end_offset);
		half4 m = (half4) (weights[matrix_offset], weights[matrix_offset + 1], weights[matrix_offset + 2], weights[matrix_offset + 3]);
		m = m * mask;
		
		const half4 v = (half4)(input[vector_offset], input[vector_offset + 1], input[vector_offset + 2], input[vector_offset + 3]);
		
		sum = mad(convert_float4(m), convert_float4(v), sum);
	}
	
	result = sum.x + sum.y + sum.z + sum.w;

#if HIDDEN_TERM
	K = HIDDEN_SIZE_X;  // width of recurrent matrix
	start_offset = GET_DATA_INDEX(RECURRENT, 0, DIRECTION, y, 0);  // set as the starting offset of the recurrent matrix 
	end_offset = start_offset + K;
	matrix_offset = start_offset + (x * 4);  // recurrent offset for the work item to work on
	vector_offset = GET_DATA_INDEX(HIDDEN, 0, 0, HIDDEN_DIRECTION, (x*4));  // hidden vector offset for the work item to work on
	sum = (float4)(0.f);
	for(; matrix_offset < end_offset; matrix_offset += (local_sz * 4), vector_offset += (local_sz * 4))
	{
		half4 mask = (half4) (1 , (matrix_offset + 1) < end_offset , (matrix_offset + 2) < end_offset , (matrix_offset + 3) < end_offset);
		half4 m = (half4) (recurrent[matrix_offset], recurrent[matrix_offset + 1], recurrent[matrix_offset + 2], recurrent[matrix_offset + 3]);
		m = m * mask;

		const half4 v = (half4) (hidden[vector_offset], hidden[vector_offset + 1], hidden[vector_offset + 2], hidden[vector_offset + 3]);
		
		sum = mad(convert_float4(m), convert_float4(v), sum);
	}
	
	result += sum.x + sum.y + sum.z + sum.w;
#endif
	
	// Add together partial sums contained in each work item's "result" variable
	SUM_ACROSS_SUB_GROUP(result);

	if(x == 0) 
	{	
	    output[y] = 0;// (half)result;

#if BIAS_TERM
		const uint bias_idx = GET_DATA_INDEX(BIAS, 0, 0, DIRECTION, y);
		half bias = biases[bias_idx];
		result += (float)bias;
#endif 

		output[y] = (half)result;
		//output[y] = convert_half_rte(result);


	}
}

#undef SUM_ACROSS_SUB_GROUP
#undef SIMD