// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/data_types.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/acc_type.cl"

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
    val += (SIMD > 8) ? intel_sub_group_shuffle(val, x+8) : 0; \
    val += (SIMD > 16) ? intel_sub_group_shuffle(val, x+16) : 0; \
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
	const int weight_num_rows = get_global_size(1);

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
		float4 mask = (float4) (1 , (matrix_offset + 1) < end_offset , (matrix_offset + 2) < end_offset , (matrix_offset + 3) < end_offset);
		float4 m = (float4) (weights[matrix_offset], weights[matrix_offset + 1], weights[matrix_offset + 2], weights[matrix_offset + 3]);
		m = m * mask;

		const float4 v = (float4) (input[vector_offset], input[vector_offset + 1], input[vector_offset + 2], input[vector_offset + 3]);

		sum = mad(m, v, sum);
	}

	result = sum.x + sum.y + sum.z + sum.w;

#if HIDDEN_TERM
	K = HIDDEN_SIZE_X;  // width of recurrent matrix
	start_offset =  GET_DATA_INDEX(RECURRENT, 0, DIRECTION, y, 0);  // set as the starting offset of the recurrent matrix
	end_offset = start_offset + K;
	matrix_offset = start_offset + (x * 4);  // recurrent offset for the work item to work on
	vector_offset = GET_DATA_INDEX(HIDDEN, 0, 0, HIDDEN_DIRECTION, (x*4));  // hidden vector offset for the work item to work on
	sum = (float4)(0.f);
	for(; matrix_offset < end_offset; matrix_offset += (local_sz * 4), vector_offset += (local_sz * 4))
	{
		float4 mask = (float4) (1 , (matrix_offset + 1) < end_offset , (matrix_offset + 2) < end_offset , (matrix_offset + 3) < end_offset);
		float4 m = (float4) (recurrent[matrix_offset], recurrent[matrix_offset + 1], recurrent[matrix_offset + 2], recurrent[matrix_offset + 3]);
		m = m * mask;

		const float4 v = (float4) (hidden[vector_offset], hidden[vector_offset + 1], hidden[vector_offset + 2], hidden[vector_offset + 3]);

		sum = mad(m, v, sum);
	}

	result += sum.x + sum.y + sum.z + sum.w;
#endif

	// Add together partial sums contained in each work item's "result" variable
	SUM_ACROSS_SUB_GROUP(result);

	if(x == 0)
	{
		output[y] = (OUTPUT_TYPE)result;

#if BIAS_TERM
		const uint bias_idx = GET_DATA_INDEX(BIAS, 0, 0, DIRECTION, y);
		float bias = (ACCUMULATOR_TYPE)biases[bias_idx];
		output[y] += (OUTPUT_TYPE)bias;
#endif
	}
}

#undef SUM_ACROSS_SUB_GROUP
#undef SIMD
