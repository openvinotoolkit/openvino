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

KERNEL(embed_ref)(const __global UNIT_TYPE* input0,
    __global UNIT_TYPE* output,
    const __global UNIT_TYPE* weights
#if BIAS_TERM
    ,const __global UNIT_TYPE* biases
#endif
)
{
    const uint x = (uint)get_global_id(0);
	const uint y = (uint)get_global_id(1);
	const uint b = (uint)get_global_id(2);

	uint output_idx = (b*INPUT0_ELEMENTS_COUNT*NUM_OUTPUT_SIZE)+(uint)(x*NUM_OUTPUT_SIZE+y);
    output[output_idx] = weights[(uint)(input0[(b*INPUT0_ELEMENTS_COUNT)+x]*NUM_OUTPUT_SIZE+y)];
#if BIAS_TERM
    output[output_idx] += biases[y];
#endif
}
