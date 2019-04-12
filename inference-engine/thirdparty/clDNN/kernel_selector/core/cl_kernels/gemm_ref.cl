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


#include "include/common.cl"
#include "include/data_types.cl"

KERNEL(gemm_ref)
	(const __global UNIT_TYPE* input0,
	const __global UNIT_TYPE* input1,
#if OUT_BIAS_TERM
	const __global UNIT_TYPE* input2,
#endif
	__global UNIT_TYPE* output)
{
    const uint x = (uint)get_global_id(0);
	const uint y = (uint)get_global_id(1);
	const uint b = (uint)get_global_id(2);
	uint in0_idx=0;
	uint in1_idx=0;
	float value = 0;
	
#if TRANSPOSE_INPUT1
for (uint i = 0; i < Y1; ++i)
	{
		in0_idx = i * X1 + x + b * X1 * Y1; 
#else
	for (uint i = 0; i < X1; ++i)
	{
		in0_idx = x * X1 + i + b * X1 * Y1; 
#endif

#if TRANSPOSE_INPUT2
	in1_idx = y * X2 + i + b * X2 * Y2;
#else
	in1_idx = i * X2 + y + b * X2 * Y2;
#endif

		value = fma(input0[in0_idx], input1[in1_idx], value);
	}
#if TRANSPOSE_INPUT1 && TRANSPOSE_INPUT2
	uint out_idx = x * Y2 + y + b * X1 * Y2;
#elif TRANSPOSE_INPUT1
	uint out_idx = x * X2 + y + b * X1 * Y1;
#elif TRANSPOSE_INPUT2
	uint out_idx = x * Y2 + y + b * X2 * Y2;
#else
	uint out_idx = x * X2 + y + b * X2 * Y1;
#endif
	
	float beta_out = 0;
#if OUT_BIAS_TERM
	beta_out = BETA * input2[out_idx];
#endif
	output[out_idx] = fma(ALPHA, value, beta_out);
}
	