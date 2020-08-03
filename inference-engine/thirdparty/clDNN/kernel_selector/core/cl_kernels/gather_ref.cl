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
#define GET_INDEX(prefix, num, idx_order) CAT(CAT(prefix, num), _GET_INDEX)(idx_order)

KERNEL(gather_ref)(const __global INPUT0_TYPE* dictionary,
                   const __global INPUT1_TYPE* indices,
                   __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
                   , FUSED_OPS_DECLS
#endif
)
{
    #if OUTPUT_DIMS == 6
        #define ORDER b,f,w,z,y,x
        const uint x = (uint)get_global_id(0) % OUTPUT_SIZE_X;
        const uint y = (uint)get_global_id(0) / OUTPUT_SIZE_X;
        const uint z = (uint)get_global_id(1) % OUTPUT_SIZE_Z;
        const uint w = (uint)get_global_id(1) / OUTPUT_SIZE_Z;
        const uint f = (uint)get_global_id(2) % OUTPUT_FEATURE_NUM;
        const uint b = (uint)get_global_id(2) / OUTPUT_FEATURE_NUM;
    #elif OUTPUT_DIMS == 5
        #define ORDER b,f,z,y,x
        const uint x = (uint)get_global_id(0);
        const uint y = (uint)get_global_id(1) % OUTPUT_SIZE_Y;
        const uint z = (uint)get_global_id(1) / OUTPUT_SIZE_Y;
        const uint f = (uint)get_global_id(2) % OUTPUT_FEATURE_NUM;
        const uint b = (uint)get_global_id(2) / OUTPUT_FEATURE_NUM;
    #elif OUTPUT_DIMS == 4
        #define ORDER b,f,y,x
        const uint x = (uint)get_global_id(0);
        const uint y = (uint)get_global_id(1);
        const uint f = (uint)get_global_id(2) % OUTPUT_FEATURE_NUM;
        const uint b = (uint)get_global_id(2) / OUTPUT_FEATURE_NUM;
    #endif

    const uint indices_idx = GET_INDICES_INDEX(INDICES_INDEX_ORDER);
    const uint dictionary_idx = GET_DICTIONARY_INDEX(DICTIONARY_INDEX_ORDER);
    const uint output_idx = GET_INDEX(OUTPUT,,ORDER);

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
