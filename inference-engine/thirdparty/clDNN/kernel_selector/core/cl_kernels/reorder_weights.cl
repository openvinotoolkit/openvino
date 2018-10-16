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


///////////////////////// Input Index /////////////////////////
inline uint FUNC(get_input_index)(uint o, uint i, uint y, uint x)
{
#if   INPUT0_SIMPLE
    return GET_FILTER_INDEX(INPUT0, o, i, y, x);
#elif defined INPUT0_LAYOUT_OS_IYX_OSV16    || \
      defined INPUT0_LAYOUT_OS_I_OSV16      || \
      defined INPUT0_LAYOUT_OS_I_OSV8__AI8  || \
      defined INPUT0_LAYOUT_OS_I_OSV16__AI8
    return GET_FILTER_OS_IYX_OSV8_INDEX(INPUT0, o, i, y, x, SUB_GROUP_SIZE);
#elif defined INPUT0_LAYOUT_OS_IYX_OSV16_ROTATE_180
    return GET_FILTER_OS_IYX_OSV8_ROTATE_180_INDEX(INPUT0, o, i, y, x, SUB_GROUP_SIZE);
#elif defined INPUT0_LAYOUT_I_YXS_OS_YXSV2_OSV16
    return GET_FILTER_I_YXS_OS_YXSV2_OSV_INDEX(INPUT0, o, i, y, x, SUB_GROUP_SIZE);
#elif defined INPUT0_LAYOUT_IY_XS_OS_XSV2_OSV16__AO32 || defined OUTPUT_LAYOUT_IY_XS_OS_XSV2_OSV8__AO32
    return GET_FILTER_IY_XS_OS_XSV2_OSV_INDEX(INPUT0, o, i, y, x, SUB_GROUP_SIZE);
#elif defined INPUT0_LAYOUT_IMAGE_2D_WEIGHTS_C1_B_FYX
    #error - not supported yet
#elif defined INPUT0_LAYOUT_OS_IS_YX_ISA8_OSV8_ISV4
	return GET_FILTER_OS_IS_YX_ISA8_OSV8_ISV4(INPUT0, o, i, y, x);
#else
#error reorder_weights.cl: input format - not supported
#endif
}

///////////////////////// Output Index /////////////////////////

inline uint FUNC(get_output_index)(uint o, uint i, uint y, uint x)
{ 
#if   OUTPUT_SIMPLE
    return GET_FILTER_INDEX(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_OS_IYX_OSV16    || \
      defined OUTPUT_LAYOUT_OS_I_OSV16      || \
      defined OUTPUT_LAYOUT_OS_I_OSV8__AI8  || \
      defined OUTPUT_LAYOUT_OS_I_OSV16__AI8
    return GET_FILTER_OS_IYX_OSV8_INDEX(OUTPUT, o, i, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_OS_IYX_OSV16_ROTATE_180
    return GET_FILTER_OS_IYX_OSV8_ROTATE_180_INDEX(OUTPUT, o, i, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_I_YXS_OS_YXSV2_OSV16
    return GET_FILTER_I_YXS_OS_YXSV2_OSV_INDEX(OUTPUT, o, i, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_IY_XS_OS_XSV2_OSV16__AO32 || defined OUTPUT_LAYOUT_IY_XS_OS_XSV2_OSV8__AO32
    return GET_FILTER_IY_XS_OS_XSV2_OSV_INDEX(OUTPUT, o, i, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_IMAGE_2D_WEIGHTS_C1_B_FYX
    return 0; //will not be used for images
#elif defined OUTPUT_LAYOUT_OS_IS_YX_ISA8_OSV8_ISV4
	return GET_FILTER_OS_IS_YX_ISA8_OSV8_ISV4(OUTPUT, o, i, y, x);
#else
#error reorder_weights.cl: output format - not supported
#endif
}

#if OUTPUT_LAYOUT_IMAGE_2D_WEIGHTS_C1_B_FYX
KERNEL (reorder_weights)(const __global INPUT0_TYPE* input, write_only image2d_t output)
{
    const unsigned o = get_global_id(0);
    const unsigned iyx = get_global_id(1);
    const unsigned x = iyx % INPUT0_SIZE_X;
    const unsigned y = (iyx / INPUT0_SIZE_X) % INPUT0_SIZE_Y;
    const unsigned i = (iyx / INPUT0_SIZE_X) / INPUT0_SIZE_Y;
    
    MAKE_VECTOR_TYPE(UNIT_TYPE, 4) input_val = (MAKE_VECTOR_TYPE(UNIT_TYPE, 4))(UNIT_VAL_ZERO, UNIT_VAL_ZERO, UNIT_VAL_ZERO, UNIT_VAL_ZERO);
    const int2 coord = (int2)(o, iyx);
    uint4 ir = FUNC_CALL(reshape_dims)(o,i,y,x, OUTPUT_SIZE_Y, OUTPUT_SIZE_X, INPUT0_SIZE_Y, INPUT0_SIZE_X, OUTPUT_DIMS, INPUT0_DIMS);
    input_val.s0 = TO_OUTPUT_TYPE(input[FUNC_CALL(get_input_index)(ir[0],ir[1],ir[2],ir[3])]);
    IMAGE_WRITE(output, coord, input_val);
}
#else
KERNEL (reorder_weights)(const __global INPUT0_TYPE* input, __global OUTPUT_TYPE* output)
{
    const unsigned o = get_global_id(0);
    const unsigned i = get_global_id(1);
#if   OUTPUT_DIMS == 2
    const unsigned y = 0;
    const unsigned x = 0;
#elif OUTPUT_DIMS == 4
    const unsigned y = get_global_id(2) / INPUT0_SIZE_X;
    const unsigned x = get_global_id(2) % INPUT0_SIZE_X;
#endif
    uint4 ir = FUNC_CALL(reshape_dims)(o,i,y,x, OUTPUT_SIZE_Y, OUTPUT_SIZE_X, INPUT0_SIZE_Y, INPUT0_SIZE_X, OUTPUT_DIMS, INPUT0_DIMS);
    output[FUNC_CALL(get_output_index)(o, i, y, x)] = TO_OUTPUT_TYPE(input[FUNC_CALL(get_input_index)(ir[0],ir[1],ir[2],ir[3])]);
}
#endif