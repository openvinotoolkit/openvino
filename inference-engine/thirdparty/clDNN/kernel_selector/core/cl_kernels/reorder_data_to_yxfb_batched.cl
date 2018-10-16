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


#include "include/reshape_dims.cl"
#include "include/fetch.cl"
#include "include/activation_functions.cl"
#include "include/data_types.cl"

///////////////////////// Input Index /////////////////////////
inline uint FUNC(get_input_index)(uint b, uint f, uint y, uint x)
{
#if   INPUT0_SIMPLE
    return GET_DATA_INDEX(INPUT0, b, f, y, x);
#elif defined INPUT0_LAYOUT_BS_F_BSV8__AF8  || \
      defined INPUT0_LAYOUT_BS_F_BSV16__AF8
    return GET_DATA_BS_FYX_BSV8_INDEX(INPUT0, b, f, y, x, SUB_GROUP_SIZE);
#elif defined INPUT0_LAYOUT_BF8_XY16
    return GET_DATA_BF8_XY16_INDEX(INPUT0, b, f, y, x);
#elif defined INPUT0_LAYOUT_BYXF_AF32
	return GET_DATA_BYXF_AF32_INDEX(INPUT0, b, f, y, x);
#else
#error reorder_data_to_yxfb_batched.cl: input format - not supported
#endif
}

inline void FUNC(get_yxfb_coords_from_linear_idx_no_padding)(uint data_idx, uint* b, uint* f, uint* x, uint* y)
{
    uint tmp_data_idx = data_idx / INPUT0_BATCH_NUM;
    *b = data_idx - tmp_data_idx * INPUT0_BATCH_NUM;
    data_idx = tmp_data_idx;

    tmp_data_idx = data_idx / INPUT0_FEATURE_NUM;
    *f = data_idx - tmp_data_idx * INPUT0_FEATURE_NUM;
    data_idx = tmp_data_idx;

    tmp_data_idx  = data_idx / INPUT0_SIZE_X;
    *x = data_idx - tmp_data_idx * INPUT0_SIZE_X;
    data_idx = tmp_data_idx;

    tmp_data_idx = data_idx / INPUT0_SIZE_Y;
    *y = data_idx - tmp_data_idx * INPUT0_SIZE_Y;
}

__attribute__((intel_reqd_sub_group_size(8)))
KERNEL (reorder_data_to_yxfb_batched)(
    const __global INPUT_REORDER_TYPE* input, 
    __global OUTPUT_REORDER_TYPE* output
    #ifdef MEAN_SUBTRACT_IN_BUFFER
    , __global MEAN_SUBTRACT_TYPE* mean_subtract
#endif
    )
{
    uint group_idx = get_group_id(0) * OUTPUT_BATCH_NUM * 8;

    for(uint i = 0; i < OUTPUT_BATCH_NUM; i++)
    {
        uint output_idx = group_idx + get_sub_group_local_id();
        if(output_idx >= ELEMENTS_COUNT)
            continue;

        group_idx += 8;

        uint x,y,f,b;
        FUNC_CALL(get_yxfb_coords_from_linear_idx_no_padding)(output_idx, &b,&f,&x,&y);
        const uint input_idx  = FUNC_CALL(get_input_index)(b, f, y, x);

    #if defined MEAN_SUBTRACT_INSIDE_PARAMS
        float res = TO_MEAN_TYPE(input[input_idx]);
        res = MEAN_OP(res, VALUE_TO_SUBTRACT[f % VALUE_TO_SUBTRACT_SIZE]);
    #elif defined MEAN_SUBTRACT_IN_BUFFER
    #if defined MEAN_PER_FEATURE
        MEAN_SUBTRACT_TYPE res = TO_MEAN_TYPE(input[input_idx]);
        res = MEAN_OP(res, mean_subtract[f]);
    #else
        MEAN_SUBTRACT_TYPE res = TO_MEAN_TYPE(input[input_idx]);
        uint4 msv = FUNC_CALL(reshape_dims)(b,f,y,x, INPUT0_SIZE_Y, INPUT0_SIZE_X, MEAN_SUBTRACT_SIZE_Y, MEAN_SUBTRACT_SIZE_X, INPUT0_DIMS, MEAN_SUBTRACT_DIMS);
        res = MEAN_OP(res, mean_subtract[GET_DATA_INDEX_SAFE(MEAN_SUBTRACT, msv[0], msv[1], msv[2], msv[3])]);
    #endif
    #else
        CALC_TYPE res = TO_CALC_TYPE(input[input_idx]);
    #endif

        output[output_idx] = ACTIVATION(TO_OUTPUT_REORDER_TYPE(res), NL_M ,NL_N);
    }
}