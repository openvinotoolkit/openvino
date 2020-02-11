/*
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
*/

#include "include/include_all.cl"

#ifdef INPUT_STRIDED
#define GET_INDEX(src) \
    GET_DATA_FS_BS_YX_BSV4_FSV32_INDEX(src, d4, d3, d2 * CAT(src, _STRIDE_Y), d1 * CAT(src, _STRIDE_X)) 
#else
#define GET_INDEX(src) \
    GET_DATA_FS_BS_YX_BSV4_FSV32_INDEX(src, d4, d3, d2, d1) 
#endif

int16 FUNC(get_int16)(const __global UNIT_TYPE* src, uint idx)
{
    int4 int_data = as_int4(intel_sub_group_block_read4((const __global uint*)(src + idx)));
    int16 to_return;
    for(uint b = 0; b < 4; b++)
    {
        for(uint f = 0; f < 4; f++)
        {
            to_return[b * 4 + f] = as_char4(int_data[b])[f];
        }
    }
    return to_return;
}
#define GET_INPUT(A, B) FUNC_CALL(get_int16)(A, GET_INDEX(B))

__attribute__((intel_reqd_sub_group_size(8)))
KERNEL(eltwise_fs_bs_yx_bsv4_fsv32)(
    INPUTS_DECLS
    __global UNIT_TYPE* output
#if CALIBRATION_TERM
    , const __global float* calibrations
#endif
    )
{
    const uint of_32_aligned = ((OUTPUT_FEATURE_NUM + 31) / 32) * 32;
    const uint d1 = get_global_id(0);   // X
    const uint d2 = get_global_id(1);   // Y
    const uint d3 = ((uint)get_global_id(2) * 4) % of_32_aligned; // Feature
    const uint d4 = 4 * (((uint)get_global_id(2) * 4) / of_32_aligned); // Batch

    int16 res;

    DO_ELTWISE;

    int4 char_result;
    for(uint b = 0; b < 4; b++)
    {
        char4 char_res;
        for(uint f = 0; f < 4; f++)
        {
            int res_tmp = res[b * 4 + f];
        #if CALIBRATION_TERM
            res_tmp = (int)round(((float)res_tmp) * calibrations[d3+f]);
        #else  // CALIBRATION_TERM
            res_tmp = (int)round(((float)res_tmp) * O_QF);
        #endif // CALIBRATION_TERM
            char_res[f] = ACTIVATION(convert_char_sat(res_tmp), ACTIVATION_PARAMS);
        }
        // pack 4 chars into int
        char_result[b] = as_int(char_res);
    }

    uint output_offset = GET_DATA_FS_BS_YX_BSV4_FSV32_INDEX(OUTPUT, d4, d3, d2, d1);
    intel_sub_group_block_write4((__global uint*)(output + output_offset), as_uint4(char_result));
}

#undef GET_INDEX
