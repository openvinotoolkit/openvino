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


#include "include/reshape_dims.cl"
#include "include/fetch.cl"

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
#elif defined INPUT0_LAYOUT_BYX8_F4
	return GET_DATA_BYX8_F4_INDEX(INPUT0, b, f, y, x);
#elif defined INPUT0_LAYOUT_FS_BS_YX_BSV4_FSV32
    return GET_DATA_FS_BS_YX_BSV4_FSV32_INDEX(INPUT0, b, f, y, x);
#elif defined INPUT0_LAYOUT_B_FS_YX_FSV4
    return GET_DATA_B_FS_YX_FSV4_INDEX(INPUT0, b, f, y, x);
#else
#error reorder_data.cl: input format - not supported
#endif
}

///////////////////////// Output Index /////////////////////////

inline uint FUNC(get_output_index)(uint b, uint f, uint y, uint x)
{
#if   OUTPUT_SIMPLE
    return GET_DATA_INDEX(OUTPUT, b, f, y, x);
#elif defined OUTPUT_LAYOUT_BS_F_BSV8__AF8  || \
      defined OUTPUT_LAYOUT_BS_F_BSV16__AF8
    return GET_DATA_BS_FYX_BSV8_INDEX(OUTPUT, b, f, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_BF8_XY16
    return GET_DATA_BF8_XY16_INDEX(OUTPUT, b, f, y, x);
#elif defined OUTPUT_LAYOUT_BYXF_AF32
	return GET_DATA_BYXF_AF32_INDEX(OUTPUT, b, f, y, x);
#elif defined OUTPUT_LAYOUT_BYX8_F4
	return GET_DATA_BYX8_F4_INDEX(OUTPUT, b, f, y, x);
#elif defined OUTPUT_LAYOUT_FS_BS_YX_BSV4_FSV32
    return GET_DATA_FS_BS_YX_BSV4_FSV32_INDEX(OUTPUT, b, f, y, x);
#elif defined OUTPUT_LAYOUT_B_FS_YX_FSV4
    return GET_DATA_B_FS_YX_FSV4_INDEX(OUTPUT, b, f, y, x);
#else
#error reorder_data.cl: output format - not supported
#endif
}

__attribute__((intel_reqd_sub_group_size(16)))
KERNEL (reorder_data_byxf_f32_to_byx8_f4_i8)(
    const __global INPUT_REORDER_TYPE* input, 
    __global OUTPUT_REORDER_TYPE* output
#ifdef MEAN_SUBTRACT_IN_BUFFER
    , __global MEAN_SUBTRACT_TYPE* mean_subtract
#endif
    )
{
    const uint x = get_global_id(0);
    const uint y = get_group_id(1);
    const uint b = get_group_id(2) * WG_BATCH_SIZE + get_sub_group_id();

    const uint input_idx  = FUNC_CALL(get_input_index)(b, 0, y, x);
    const uint output_idx = FUNC_CALL(get_output_index)(b, 0, y, x);

#if defined MEAN_SUBTRACT_INSIDE_PARAMS
    float4 res;
    res.s0 = TO_MEAN_TYPE(input[input_idx]);
    res.s0 = MEAN_OP(res.s0, VALUE_TO_SUBTRACT[0 % VALUE_TO_SUBTRACT_SIZE]);
    res.s1 = TO_MEAN_TYPE(input[input_idx+1]);
    res.s1 = MEAN_OP(res.s1, VALUE_TO_SUBTRACT[1 % VALUE_TO_SUBTRACT_SIZE]);
    res.s2 = TO_MEAN_TYPE(input[input_idx+2]);
    res.s2 = MEAN_OP(res.s2, VALUE_TO_SUBTRACT[2 % VALUE_TO_SUBTRACT_SIZE]);
    res.s3 = 0;
#elif defined MEAN_SUBTRACT_IN_BUFFER
#if defined MEAN_PER_FEATURE
    MAKE_VECTOR_TYPE(MEAN_SUBTRACT_TYPE, 4) res;
    res.s0 = TO_MEAN_TYPE(input[input_idx]);
    res.s0 = MEAN_OP(res.s0, mean_subtract[0]);
    res.s1 = TO_MEAN_TYPE(input[input_idx+1]);
    res.s1 = MEAN_OP(res.s1, mean_subtract[1]);
    res.s2 = TO_MEAN_TYPE(input[input_idx+2]);
    res.s2 = MEAN_OP(res.s2, mean_subtract[2]);
    res.s3 = 0
#else
    MAKE_VECTOR_TYPE(MEAN_SUBTRACT_TYPE, 4) res;
    res.s0 = TO_MEAN_TYPE(input[input_idx]);
    res.s1 = TO_MEAN_TYPE(input[input_idx+1]);
    res.s2 = TO_MEAN_TYPE(input[input_idx+2]);
    res.s3 = 0; 

    res.s0 = MEAN_OP(res.s0, mean_subtract[0]);
    res.s1 = MEAN_OP(res.s1, mean_subtract[1]);
    res.s2 = MEAN_OP(res.s2, mean_subtract[2]);
#endif
#else
    MAKE_VECTOR_TYPE(CALC_TYPE, 4) res;
    res.s0 = TO_CALC_TYPE(input[input_idx]);
    res.s1 = TO_CALC_TYPE(input[input_idx+1]);
    res.s2 = TO_CALC_TYPE(input[input_idx+2]);
    res.s3 = 0;
#endif

    char4 out_vals;
    out_vals.s0 = ACTIVATION_TYPED(OUTPUT_REORDER, TO_OUTPUT_REORDER_TYPE_SAT(res.s0), ACTIVATION_PARAMS_TYPED);
    out_vals.s1 = ACTIVATION_TYPED(OUTPUT_REORDER, TO_OUTPUT_REORDER_TYPE_SAT(res.s1), ACTIVATION_PARAMS_TYPED);
    out_vals.s2 = ACTIVATION_TYPED(OUTPUT_REORDER, TO_OUTPUT_REORDER_TYPE_SAT(res.s2), ACTIVATION_PARAMS_TYPED);
    out_vals.s3 = 0;

    __global uint* dst = (__global uint*)output;
    dst[output_idx/4] = as_uint(out_vals);
}
