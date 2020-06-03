// Copyright (c) 2019-2020 Intel Corporation
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

#define INPUT_AXIS_INDEX (uint)indices[indices_idx]
#define GET_DICTIONARY_INDEX(idx_order) INPUT0_GET_INDEX(idx_order)
#define GET_INDICES_INDEX(idx_order) INPUT1_GET_INDEX(idx_order)

KERNEL(gather_ref)(const __global INPUT0_TYPE* dictionary,
                   const __global INPUT1_TYPE* indices,
                   __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
                   , FUSED_OPS_DECLS
#endif
)
{
    const uint b = get_global_id(0);
    const uint f = get_global_id(1);
    const uint yx = get_global_id(2);
    const uint y = yx / OUTPUT_SIZE_X;
    const uint x = yx % OUTPUT_SIZE_X;

    const uint indices_idx = GET_INDICES_INDEX(INDICES_INDEX_ORDER);
    const uint dictionary_idx = GET_DICTIONARY_INDEX(DICTIONARY_INDEX_ORDER);
    const uint output_idx = OUTPUT_GET_INDEX(b, f, y, x);

    INPUT0_TYPE val = dictionary[dictionary_idx];

#if HAS_FUSED_OPS
    FUSED_OPS;
    output[output_idx] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT);
#else
    output[output_idx] = ACTIVATION(val, ACTIVATION_PARAMS);
#endif
}

#undef GET_INDICES_INDEX
#undef GET_DICTIONARY_INDEX
#undef INPUT_AXIS_INDEX
